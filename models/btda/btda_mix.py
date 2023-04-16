import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import os
import contextlib

#from models.flow.glow import Glow
from train_utils import AverageMeter, calc_loss

from .linear_utils import  Get_Scalar, one_hot, mixup_one_target
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller

from sklearn.metrics import *
import numpy as np
import json
from copy import deepcopy


class Linear:
    def __init__(self, net_builder, num_classes, ema_m, T, lambda_u, \
                 num_eval_iter=1000, tb_log=None, logger=None):
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

        super(Linear, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        self.model = net_builder(num_classes=num_classes)

        self.inverse_matrix = nn.Linear(in_features=num_classes, out_features=num_classes, bias=False).cuda()

        self.ema_model = deepcopy(self.model)

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.lambda_u = lambda_u
        self.tb_log = tb_log

        self.optimizer = None
        self.scheduler = None

        # self.optimizer_matrix = torch.optim.SGD(self.inverse_matrix.parameters(), lr=0.003, momentum=0.9,
        #                                         weight_decay=0.0005)
        self.optimizer_linear = torch.optim.Adam(self.inverse_matrix.parameters(), lr=0.003)

        self.it = 0
        self.iter_flow = 20000
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    '''
    def train_iter(self, args, x_lb, x_ulb_w):
        # inference and calculate sup/unsup losses
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext
        with amp_cm():
            with torch.no_grad():
                #self.bn_controller.freeze_bn(self.model)
                logits_x_lb = self.model(x_lb)
                logits_x_ulb_w = self.model(x_ulb_w)
                #self.bn_controller.unfreeze_bn(self.model)

                # prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)
                # prob_x_lb = torch.softmax(logits_x_lb, dim=1)
                prob_x = torch.cat([logits_x_lb, logits_x_ulb_w], dim=0)

            out_linear = self.inverse_matrix(prob_x)

            # logdet = logdet.mean()

            # total_loss, log_p, log_det = calc_loss(log_p, logdet, x_lb.shape[2])
            # print('flow: loss:{} log_p:{}, logdet:{}'.format(total_loss.item(), log_p.item(), log_det.item()))

        # self.inverse_matrix.zero_grad()
        # if args.amp:
        #     scaler.scale(total_loss).backward()
        #     if (args.clip > 0):
        #         torch.nn.utils.clip_grad_norm_(self.flow.parameters(), args.clip)
        #     scaler.step(self.optimizer_flow)
        #     scaler.update()
        # else:
        #     total_loss.backward()
        #     if (args.clip > 0):
        #         torch.nn.utils.clip_grad_norm_(self.flow.parameters(), args.clip)
        #     self.optimizer_flow.step()

    def train_flow(self, args, nums):

        it_flow = 0
        # x_ulb_s1_rot: rotated data, rot_v: rot angles
        for (_, x_lb, y_lb), (_, x_ulb_w, x_ulb_s1, x_ulb_s2, x_ulb_s1_rot, rot_v) in zip(self.loader_dict['train_lb'],
                                                                                          self.loader_dict[
                                                                                              'train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if it_flow > nums:
                break

            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s1.shape[0]

            x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s1.cuda(
                args.gpu), x_ulb_s2.cuda(args.gpu)
            self.train_iter(args, x_lb, x_ulb_w)
            it_flow += 1

    '''
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
        print('p_target:', p_target)

        p_model = None
        q_model = None


        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)


        # x_ulb_s1_rot: rotated data, rot_v: rot angles
        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w1, x_ulb_w2) in zip(self.loader_dict['train_lb'],
                                                                  self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w1.shape[0]
            assert num_ulb == x_ulb_w2.shape[0]

            y_lb = torch.zeros(args.batch_size, args.num_classes).scatter_(1, y_lb.view(-1, 1), 1)
            x_lb, x_ulb_w1, x_ulb_w2 = x_lb.cuda(args.gpu), x_ulb_w1.cuda(args.gpu), x_ulb_w2.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            # inference and calculate sup/unsup losses
            with amp_cm():
                #self.train_iter(args, x_lb, x_ulb_w)
                with torch.no_grad():

                    logits_x_lb = self.model(x_lb)
                    logits_x_ulb_w1 = self.model(x_ulb_w1)
                    logits_x_ulb_w2 = self.model(x_ulb_w2)

                    # hyper-params for update
                    T = self.t_fn(self.it)

                    prob_x_ulb = torch.softmax(logits_x_ulb_w1, dim=1)
                    prob_x_lb = torch.softmax(logits_x_lb, dim=1)

                        # p^~_(y): moving average of p(y)
                    if p_model == None:
                        p_model = torch.mean(prob_x_ulb.detach(), dim=0)
                    else:
                        p_model = p_model * 0.9 + torch.mean(prob_x_ulb.detach(), dim=0) * 0.1
                    if q_model == None:
                        q_model = torch.mean(prob_x_lb.detach(), dim=0)
                    else:
                        q_model = q_model * 0.9 + torch.mean(prob_x_lb.detach(), dim=0) * 0.1

                    prob_x_ulb_t = prob_x_ulb * q_model / p_model

                logits_x_ulb_w_p = self.inverse_matrix(logits_x_ulb_w1)
                linear_loss = F.kl_div(torch.softmax(logits_x_ulb_w_p, dim=1).log(), prob_x_ulb_t)

                # parameter updates
                if args.amp:
                    scaler.scale(linear_loss).backward()
                    if (args.clip > 0):
                        torch.nn.utils.clip_grad_norm_(self.inverse_matrix.parameters(), args.clip)
                    scaler.step(self.optimizer_linear)
                    scaler.update()
                else:
                    linear_loss.backward()
                    if (args.clip > 0):
                        torch.nn.utils.clip_grad_norm_(self.inverse_matrix.parameters(), args.clip)
                    self.optimizer_linear.step()

                with torch.no_grad():
                
                    logits_x_ulb_w1 = self.inverse_matrix(logits_x_ulb_w1)
                    logits_x_ulb_w2 = self.inverse_matrix(logits_x_ulb_w2)
                    prob_x_ulb = (torch.softmax(logits_x_ulb_w1, dim=1)+torch.softmax(logits_x_ulb_w2, dim=1))/2
                    prob_x_ulb = (prob_x_ulb /prob_x_ulb.sum(dim=-1, keepdim=True))
                    prob_x_ulb=prob_x_ulb**(1/T)
                    prob_x_ulb = (prob_x_ulb / prob_x_ulb.sum(dim=-1, keepdim=True))

                    # mix up
                    inputs = torch.cat((x_lb,x_ulb_w1,x_ulb_w2))
                    targets = torch.cat((y_lb, prob_x_ulb, prob_x_ulb))
                    mixed_x, mixed_y, _ = mixup_one_target(inputs, targets,
                                                           args.gpu,
                                                           args.alpha,
                                                           is_bias=True)
                    mixed_x = list(torch.split(mixed_x, num_lb))
                    mixed_x = self.interleave(mixed_x, num_lb)



                logits = [self.model(mixed_x[0])]

                for input in mixed_x[1:]:
                    logits.append(self.model(input))

                # put interleaved samples back
                logits = self.interleave(logits, num_lb)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)

                # semiloss
                probs_u = torch.softmax(logits_u, dim=1)
                Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_y[:num_lb], dim=1))
                Lu = torch.mean((probs_u - mixed_y[num_lb:]) ** 2)

                rampup = float(np.clip(self.it / (args.ramp_up * args.num_train_iter), 0.0, 1.0))
                lambda_u = self.lambda_u * rampup

                total_loss = Lx + lambda_u * Lu


            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                #scaler.scale(linear_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    #torch.nn.utils.clip_grad_norm_(self.inverse_matrix.parameters(), args.clip)
                scaler.step(self.optimizer)
                #scaler.step(self.optimizer_linear)
                scaler.update()
            else:
                total_loss.backward()
                #linear_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    #torch.nn.utils.clip_grad_norm_(self.inverse_matrix.parameters(), args.clip)
                self.optimizer.step()
                #self.optimizer_linear.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()
            self.inverse_matrix.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = Lx.detach()
            tb_dict['train/unsup_loss'] = Lu.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            #tb_dict['train/mask_ratio'] = 1.0 - select_mask.mean().detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                eval_dict1=eval_dict.copy()
                eval_dict1.pop('eval/recall_each')
                tb_dict.update(eval_dict1)
                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    recall_best=eval_dict['eval/recall_each']
                    recall=eval_dict['eval/recall']
                    best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, RECALL:{recall_best},RECALL_avg:{recall} at {best_it} iters")
                total_time = 0

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            #print(self.it)
            del tb_dict
            start_batch.record()
            #if self.it > 0.8 * args.num_train_iter:
            #    self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it ,'ecal/recall_each':recall_best})
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
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        recall_each,recall=self.get_recall_each(y_true,y_pred,args.num_classes)
        top1 = accuracy_score(y_true, y_pred,)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5 ,'eval/recall_each':recall_each,'eval/recall':recall}

    def get_recall_each(self,y_true,y_pred,num_classes):
        cur = torch.zeros(num_classes).cuda()
        y_true=torch.tensor(y_true).cuda()
        y_pred=torch.tensor(y_pred).cuda()
        for cl in torch.unique(y_true):
            y_t= y_true==cl
            eq= y_true==y_pred
            cur[cl]=torch.sum(y_t&eq)
        class_count = torch.tensor([(y_true == t).sum() for t in torch.unique(y_true, sorted=True)]).cuda()
        recall_each=cur/class_count
        recall = torch.mean(recall_each)
        return recall_each,recall


    def save_model(self, save_name, save_path):

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
