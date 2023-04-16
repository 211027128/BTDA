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

class MyLinear(nn.Linear):
    def forward(self, input: torch.Tensor, u2l=True) -> torch.Tensor:
        if u2l:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, torch.inverse(self.weight), self.bias).clamp(-5, 5)

class Linear:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff,lambda_u, \
                 hard_label=True,
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

        #self.inverse_matrix = nn.Linear(in_features=num_classes, out_features=num_classes, bias=False).cuda()
        self.inverse_matrix = MyLinear(in_features=num_classes, out_features=num_classes, bias=False).cuda()

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
        p_std = None
        q_std = None


        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        # x_ulb_s1_rot: rotated data, rot_v: rot angles
        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                                                                  self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            y_lb = torch.zeros(args.batch_size, args.num_classes).scatter_(1, y_lb.view(-1, 1), 1)
            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            batch_size = x_ulb_w.size(0)
            # inference and calculate sup/unsup losses
            with amp_cm():
                #self.train_iter(args, x_lb, x_ulb_w)
                with torch.no_grad():

                    logits_x_lb = self.model(x_lb)
                    logits_x_ulb_w = self.model(x_ulb_w)

                    prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)
                    prob_x_lb = torch.softmax(logits_x_lb, dim=1)

                    # p^~_(y): moving average of p(y)
                    if p_model == None:
                        p_model = torch.mean(prob_x_ulb.detach(), dim=0)
                        p_std = 1e-5
                    else:
                        p_model = p_model * 0.9 + torch.mean(prob_x_ulb.detach(), dim=0) * 0.1
                        cur_p_std = torch.sqrt(torch.mean((prob_x_ulb.detach() - p_model) ** 2, dim=0))
                        p_std = p_std * 0.9 + cur_p_std * 0.1

                    if q_model == None:
                        q_model = torch.mean(prob_x_lb.detach(), dim=0)
                        q_std = 1e-5
                    else:
                        q_model = q_model * 0.9 + torch.mean(prob_x_lb.detach(), dim=0) * 0.1
                        cur_q_std = torch.sqrt(torch.mean((prob_x_lb.detach() - q_model) ** 2, dim=0))
                        q_std = q_std * 0.9 + cur_q_std * 0.1

                    # ------sample pair
                    prob_x_ulb_tran = (prob_x_ulb - p_model) / p_std * q_std + q_model
                    prob_x_lb_tran = (prob_x_lb - q_model) / q_std * p_std + p_model

                pseudo_q_model = self.inverse_matrix(logits_x_ulb_w)
                logits_x_lb_t = self.inverse_matrix(logits_x_lb, u2l=False)
                linear_loss = 0.5 * F.kl_div(torch.softmax(pseudo_q_model, dim=1).log(), prob_x_ulb_tran) \
                              + 0.5 * F.kl_div(torch.softmax(logits_x_lb_t, dim=1).log(), prob_x_lb_tran)

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
                    logits_x_ulb_ww = self.inverse_matrix(logits_x_ulb_w)
                    if self.it >=50000:
                        prob_x_ulb = torch.softmax(logits_x_ulb_ww, dim=1)

                    max_p, p_hat = torch.max(prob_x_ulb, dim=1)
                    p_hat = torch.zeros(batch_size, args.num_classes).cuda().scatter_(1, p_hat.view(-1, 1), 1)

                    select_mask = max_p.ge(args.p_cutoff)
                    select_mask = torch.cat([select_mask, select_mask], 0).float()

                    # mix up
                    inputs = torch.cat((x_lb,x_ulb_w,x_ulb_s))
                    targets = torch.cat((y_lb, p_hat, p_hat))

                logits = self.model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ub = logits[num_lb:]
                Lx = -torch.mean(torch.sum(F.log_softmax(logits_x_lb, dim=1) * targets[:num_lb], dim=1))
                Lu = -torch.mean(torch.sum(F.log_softmax(logits_x_ub, dim=1) * targets[num_lb:], dim=1) * select_mask)
                total_loss = Lx + self.lambda_u * Lu

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
            self.inverse_matrix.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = Lx.detach()
            tb_dict['train/unsup_loss'] = Lu.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - select_mask.mean().detach()
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

                    #if self.it == best_it:
                    #    self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            #print(self.it,self.iter_flow)
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
        #self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
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
        recall = torch.mean(cur)
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
