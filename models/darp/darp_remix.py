import pickle
import random
import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import contextlib
from train_utils import AverageMeter

from .darp_utils import consistency_loss, Get_Scalar, estimate_pseudo, opt_solver, interleave, linear_rampup, \
    mixup_one_target
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller, SemiLoss

from sklearn.metrics import *
from copy import deepcopy


class Darp:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
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

        super(Darp, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = None

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
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

    def train(self, args, emp_distb_u,target_disb, pseudo_orig, pseudo_refine, logger=None):

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

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)


        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s1, x_ulb_s2, x_ulb_s1_rot, rot_v) in zip(self.loader_dict['train_lb'],
                                                                                          self.loader_dict[
                                                                                              'train_ulb']):
            star = time.time()
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            num_rot = x_ulb_s1_rot.shape[0]
            assert num_ulb == x_ulb_s1.shape[0]
            
            #print(rot_v.size(),y_lb.size())

            x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s1.cuda(
                args.gpu), x_ulb_s2.cuda(args.gpu)
            x_ulb_s1_rot = x_ulb_s1_rot.cuda(args.gpu)  # rot_image
            rot_v = torch.zeros(args.batch_size, 4).scatter_(1, rot_v.view(-1, 1), 1)
            rot_v = rot_v.cuda(args.gpu)  # rot_label
            y_lb = torch.zeros(args.batch_size, args.num_classes).scatter_(1, y_lb.view(-1, 1), 1)
            y_lb = y_lb.cuda(args.gpu)


            batch_size = y_lb.size(0)

            # Generate the pseudo labels
            with torch.no_grad():
                # Generate the pseudo labels by aggregation and sharpening
                #self.bn_controller.freeze_bn(self.model)
                outputs_u = self.model(x_ulb_w)[0]
                #self.bn_controller.unfreeze_bn(self.model)
                p = torch.softmax(outputs_u, dim=1)

                #if self.it == 0:
                #    emp_distb_u = p.mean(0, keepdim=True)
                #elif self.it // 128 == 0:
                #    emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)
                #else:
                #    emp_distb_u = emp_distb_u[-127:]
                #    emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)

                #pa = p * (target_disb.cuda() + 1e-6) / (emp_distb_u.mean(0).cuda() + 1e-6)
                #p = pa / pa.sum(dim=1, keepdim=True)

                pt=p**(1 / args.T)
                targets_u = (pt / pt.sum(dim=1, keepdim=True)).detach()

                # Update the saved predictions with current one
                p = targets_u
                pseudo_orig[x_ulb_idx, :] = p.data.cpu()
                pseudo_orig_backup = pseudo_orig.clone()

                # Applying DARP
                # if True:   #test darp
                if self.it > args.num_train_iter * 3 / 5 :
                #if self.it > 0:
                    if self.it % args.num_iter == 0:
                        # Iterative normalization
                        targets_u, weights_u = estimate_pseudo(target_disb, pseudo_orig, args.num_classes,
                                                               alpha=args.uratio)
                        scale_term = targets_u * weights_u.reshape(1, -1)
                        pseudo_orig = (pseudo_orig * scale_term + 1e-6) / (pseudo_orig * scale_term + 1e-6).sum(dim=1,
                                                                                                                keepdim=True)

                        if args.dataset == 'stl10' or args.dataset == 'cifar100':
                            opt_res = opt_solver(pseudo_orig, target_disb, args.iter_T, 0.3)
                        else:
                            opt_res = opt_solver(pseudo_orig, target_disb, args.iter_T, 0.1)

                        # Updated pseudo-labels are saved
                        pseudo_refine = opt_res

                        # Select
                        targets_u = opt_res[x_ulb_idx].detach().cuda()
                        pseudo_orig = pseudo_orig_backup
                    else:
                        # Using previously saved pseudo-labels
                        targets_u = pseudo_refine[x_ulb_idx].cuda()

            # Mixup
            all_inputs = torch.cat([x_lb, x_ulb_w, x_ulb_s1,x_ulb_s2], dim=0)
            all_targets = torch.cat([y_lb, targets_u, targets_u,targets_u], dim=0)

            mixed_input, mixed_target, _ = mixup_one_target(all_inputs, all_targets,
                                                            args.gpu,
                                                            args.alpha,
                                                            is_bias=True)

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)



            # inference and calculate sup/unsup losses
            with amp_cm():
                logits = [self.model(mixed_input[0])[0]]
                for input in mixed_input[1:]:
                    logits.append(self.model(input)[0])

                # put interleaved samples back
                logits = interleave(logits, batch_size)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:])
                logits_rot = self.model(x_ulb_s1_rot)[1]
                u1_logits = self.model(x_ulb_s1)[0]

                #semiloss
                Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size], dim=1))
                Lu = -torch.mean(torch.sum(F.log_softmax(logits_u, dim=1) * mixed_target[batch_size:], dim=1))
                Lr = -torch.mean(torch.sum(F.log_softmax(logits_rot, dim=1) * rot_v, dim=1))
                Le = -torch.mean(torch.sum(F.log_softmax(u1_logits, dim=1) * targets_u, dim=1))

                w_match = args.w_match * float(np.clip(self.it / (args.warm_up * args.num_train_iter), 0.0, 1.0))
                w_kl = args.w_kl * float(np.clip(self.it / (args.warm_up * args.num_train_iter), 0.0, 1.0))


                total_loss = Lx + w_match*Lu + Lr*args.w_rot+w_kl * Le

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
            tb_dict['train/sup_loss'] = Lx.detach()
            tb_dict['train/unsup_loss'] = Lu.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 10K steps and best model for each 1K steps
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
                    f"DARP=== {self.it} iteration,{tb_dict}, recal:{recall_best},BEST_EVAL_ACC: {best_eval_acc},at {best_it} iters")
                total_time = 0

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            end=time.time()
            #print(self.it,end-star)
            del tb_dict
            start_batch.record()
            #if self.it > 0.8 * args.num_train_iter:
            #    self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
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
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        #self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5 ,'eval/recall_each':recall_each}

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

