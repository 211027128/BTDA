import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import os
import contextlib
from train_utils import AverageMeter

from .adsh_utils import consistency_loss, Get_Scalar, one_hot, mixup_one_target
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller

from sklearn.metrics import *
import numpy as np
import json
from copy import deepcopy


class Adsh:
    def __init__(self, net_builder, num_classes, ema_m, T, lambda_u, \
                 w_match,
                 t_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(Adsh, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        self.model = net_builder(num_classes=num_classes)
        self.ema_model = deepcopy(self.model)

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.w_match = w_match  # weight of distribution matching
        self.lambda_u = lambda_u
        self.tb_log = tb_log

        self.optimizer = None
        self.scheduler = None

        self.it = 0

        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def update_s(self, args, score, unlabel_loader):
        with torch.no_grad():
            lists = [[] for _ in range(args.num_classes)]
            for batch_idx, (_, inputs_uw, inputs_us, _, _, _) in enumerate(unlabel_loader):
                self.model.eval()
                inputs_uw = inputs_uw.cuda(args.gpu)
                outputs = self.model(inputs_uw)[0]
                logits = torch.softmax(outputs.detach(), dim=1)
                max_probs, targets_u = torch.max(logits, dim=1)

                for i in range(targets_u.shape[0]):
                    lists[targets_u[i]].append(max_probs[i].detach().cpu().numpy())

            lists = np.array(lists, dtype=object)
            rho = 1.0
            for i in range(lists.shape[0]):
                lists[i] = np.sort(np.array(lists[i]))[::-1]
            for i in range(lists[0].shape[0]):
                if lists[0][i] < 0.95:
                    break
                rho = (i + 1) / lists[0].shape[0]
            for i in range(1, lists.shape[0]):
                if lists[i].shape[0] != 0:
                    idx = max(0, np.round(lists[i].shape[0] * rho - 1).astype(int))
                    score[i] = min(0.95, lists[i][idx])
        return score
    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # p(y) based on the labeled examples seen during training
        dist_file_name = r"./data_statistics/" + args.dataset + '_' + str(args.num_labels) + '.json'
        with open(dist_file_name, 'r') as f:
            p_target = json.loads(f.read())
            p_target = torch.tensor(p_target['distribution'])
            p_target = p_target.cuda(args.gpu)

        # p_target=torch.tensor([0.0613, 0.0829, 0.0493, 0.1311, 0.0768, 0.1457, 0.0500, 0.0800, 0.0799,0.2429]).cuda(args.gpu)
        print('p_target:', p_target)

        p_model = None
        score = np.zeros(args.num_classes) + 0.95

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        # x_ulb_s1_rot: rotated data, rot_v: rot angles
        for (_, x_lb, y_lb), (_, x_ulb_w, x_ulb_s1, x_ulb_s2, x_ulb_s1_rot, rot_v) in zip(self.loader_dict['train_lb'],
                                                                                          self.loader_dict[
                                                                                              'train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            print(self.it)
            if self.it > 1:
                score = self.update_s(args, score, self.loader_dict['train_ulb'])

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            num_rot = x_ulb_s1_rot.shape[0]
            assert num_ulb == x_ulb_s1.shape[0]

            x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s1.cuda(
                args.gpu), x_ulb_s2.cuda(args.gpu)
            x_ulb_s1_rot = x_ulb_s1_rot.cuda(args.gpu)  # rot_image
            rot_v = rot_v.cuda(args.gpu)  # rot_label
            y_lb = y_lb.cuda(args.gpu)

            # inference and calculate sup/unsup losses
            with amp_cm():
                with torch.no_grad():
                    # self.bn_controller.freeze_bn(self.model)
                    # logits_x_lb = self.model(x_lb)[0]
                    logits_x_ulb_w = self.model(x_ulb_w)[0]
                    logits_x_ulb_s1 = self.model(x_ulb_s1)[0]
                    # logits_x_ulb_s2 = self.model(x_ulb_s2)[0]
                    # self.bn_controller.unfreeze_bn(self.model)

                    # hyper-params for update
                    T = self.t_fn(self.it)

                    prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)

                    # p^~_(y): moving average of p(y)
                    '''
                    if p_model == None:
                        p_model = torch.mean(prob_x_ulb.detach(), dim=0)
                    else:
                        p_model = p_model * 0.999 + torch.mean(prob_x_ulb.detach(), dim=0) * 0.001

                    prob_x_ulb = prob_x_ulb * p_target / p_model
                    prob_x_ulb = (prob_x_ulb / prob_x_ulb.sum(dim=-1, keepdim=True))
                    '''

                    sharpen_prob_x_ulb = prob_x_ulb ** (1 / T)
                    sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

                    # mix up
                    mixed_inputs = torch.cat((x_lb, x_ulb_s1, x_ulb_s2, x_ulb_w))
                    input_labels = torch.cat(
                        [one_hot(y_lb, args.num_classes, args.gpu), sharpen_prob_x_ulb, sharpen_prob_x_ulb,
                         sharpen_prob_x_ulb], dim=0)

                    mixed_x, mixed_y, _ = mixup_one_target(mixed_inputs, input_labels,
                                                           args.gpu,
                                                           args.alpha,
                                                           is_bias=True)

                    # Interleave labeled and unlabeled samples between batches to get correct batch norm calculation
                    mixed_x = list(torch.split(mixed_x, num_lb))
                    mixed_x = self.interleave(mixed_x, num_lb)

                    # inter_inputs = torch.cat([mixed_x, x_ulb_s1], dim=0)
                    # inter_inputs = list(torch.split(inter_inputs, num_lb))
                    # inter_inputs = self.interleave(inter_inputs, num_lb)

                    # calculate BN only for the first batch
                logits = [self.model(mixed_x[0])[0]]

                # self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt)[0])

                u1_logits = self.model(x_ulb_s1)[0]
                logits_rot = self.model(x_ulb_s1_rot)[1]
                logits = self.interleave(logits, num_lb)
                # self.bn_controller.unfreeze_bn(self.model)

                logits_x = logits[0]
                logits_u = torch.cat(logits[1:])

                probs = torch.softmax(logits_u, dim=1)
                rectify_prob = probs / torch.from_numpy(score).float().cuda(args.gpu)
                max_rp, rp_hat = torch.max(rectify_prob, dim=1)
                mask = max_rp.ge(1.0)

                # calculate rot loss with w_rot
                rot_loss = ce_loss(logits_rot, rot_v, reduction='mean')
                rot_loss = rot_loss.mean()
                # sup loss
                sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False)
                sup_loss = sup_loss.mean()
                # unsup_loss
                unsup_loss = ce_loss(logits_u, mixed_y[num_lb:], use_hard_labels=False) * mask
                unsup_loss = unsup_loss.mean()
                # loss U1
                u1_loss = ce_loss(u1_logits, sharpen_prob_x_ulb, use_hard_labels=False)
                u1_loss = u1_loss.mean()
                # ramp for w_match
                w_match = args.w_match * float(np.clip(self.it / (args.warm_up * args.num_train_iter), 0.0, 1.0))
                w_kl = args.w_kl * float(np.clip(self.it / (args.warm_up * args.num_train_iter), 0.0, 1.0))

                total_loss = sup_loss + args.w_rot * rot_loss + w_kl * u1_loss + w_match * unsup_loss
                # total_loss = sup_loss + w_kl * u1_loss + w_match * unsup_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 100000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                eval_dict1 = eval_dict.copy()
                eval_dict1.pop('eval/recall_each')
                tb_dict.update(eval_dict1)

                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    recall_best = eval_dict['eval/recall_each']
                    best_it = self.it

                self.print_fn(
                    f"Adsh=== {self.it} iteration,{tb_dict}, recal:{recall_best},BEST_EVAL_ACC: {best_eval_acc},at {best_it} iters")
                total_time = 0

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    # if self.it == best_it:
                    #    self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            # if self.it > 0.8 * args.num_train_iter:
            #    self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it, 'ecal/recall_each': recall_best})
        self.print_fn(eval_dict)
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits, _ = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        recall_each = self.get_recall_each(y_true, y_pred, args.num_classes)
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        # cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        # self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/recall_each': recall_each}

    def get_recall_each(self, y_true, y_pred, num_classes):
        cur = torch.zeros(num_classes).cuda()
        y_true = torch.tensor(y_true).cuda()
        y_pred = torch.tensor(y_pred).cuda()
        for cl in torch.unique(y_true):
            y_t = y_true == cl
            eq = y_true == y_pred
            cur[cl] = torch.sum(y_t & eq)
        class_count = torch.tensor([(y_true == t).sum() for t in torch.unique(y_true, sorted=True)]).cuda()
        recall_each = cur / class_count
        return recall_each

    def save_model(self, save_name, save_path):
        if self.it < 1000000:
            return
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = deepcopy(self.model)
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model.state_dict()},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.print_fn('model loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass