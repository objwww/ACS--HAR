import pickle


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

from .ACS_HAR_utils import entropy_loss, consistency_loss, Get_Scalar, focal_loss
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller,EarlyStopping
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from copy import deepcopy
import time
from torch.utils.tensorboard import SummaryWriter

class ACS_HAR:
    def __init__(self, net_builder, num_classes, ema_m, lambda_u, lambda_e, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Freematch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            ema_m: momentum of exponential moving average for eval_model
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(ACS_HAR, self).__init__()

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
        self.lambda_u = lambda_u
        self.lambda_e = lambda_e
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.epoch=0
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()
        self.earlyStopping=EarlyStopping(patience=60,verbose=True)
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def warmup(self, args, logger=None):
        ngpus_per_node = torch.cuda.device_count()

        self.model.train()
        warmup_it = 0
    
        writer=SummaryWriter(os.path.join(self.tb_dir, file_name))
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext
        start_epoch=AverageMeter()
        end_epoch=AverageMeter()
        end=time.time()
        sup_loss_meter=AverageMeter()
        for _ in range (0,args.epoch):
            sup_loss_meter.reset()
            start_epoch.update(time.time()-end)
            if warmup_it>200:
                break
            for _, x_lb, y_lb in self.loader_dict['train_lb']:           
                x_lb = x_lb.cuda(args.gpu)
                y_lb = y_lb.cuda(args.gpu)

                num_lb = x_lb.shape[0]

                with amp_cm():

                    logits_x_lb,feature = self.model(x_lb)
                    sup_loss = ce_loss(logits_x_lb, y_lb,use_hard_labels=True, reduction='mean')

                    total_loss = sup_loss

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

                self.model.zero_grad()
                # tensorboard_dict update
            sup_loss_meter.update(sup_loss.item())
            end_epoch.update(time.time()-end)
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss_meter.avg
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_epoch.avg/ 1000.
            tb_dict['train/run_time'] = end_epoch.avg/ 1000.

            if warmup_it % 50 == 0:
                self.print_fn(f"warmup {warmup_it} iteration, {tb_dict}")

            del tb_dict
           
            warmup_it += 1

        # compute p_model, time_p, 
        self.model.eval()
        probs = []
        with torch.no_grad():
            for _, x, y in self.loader_dict['eval']:

                x = x.cuda(args.gpu)
                y = y.cuda(args.gpu)

                # inference and calculate sup/unsup losses
                with amp_cm():
                    logits,feature = self.model(x)
                probs.append(logits.softmax(dim=-1).cpu())
            
        probs = torch.cat(probs)
        max_probs, max_idx = torch.max(probs, dim=-1)

        self.time_p = max_probs.mean()
        self.p_model = torch.mean(probs, dim=0)
        label_hist = torch.bincount(max_idx, minlength=probs.shape[1]).to(probs.dtype) 
        self.label_hist = label_hist / label_hist.sum()

    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count()
        
        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)
        # start_batch.record()
        best_eval_acc, best_it = 0.0, 0
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)
        p_model = (torch.ones(args.num_classes) / args.num_classes).cuda()
        label_hist = (torch.ones(args.num_classes) / args.num_classes).cuda() 
        time_p = p_model.mean()
        sup_loss_meter=AverageMeter()
        total_loss_meter=AverageMeter()
        start_epoch_meter=AverageMeter()
        run_meter=AverageMeter()
        ps_label_hist_meter=AverageMeter()
        label_hist_meter=AverageMeter()
        p_model_meter=AverageMeter()
        mask_ratio_meter=AverageMeter()
        time_p_meter=AverageMeter()
        ent_loss_meter=AverageMeter()
        unsup_loss_meter=AverageMeter()
        # eval for once to verify if the checkpoint is loaded correctly
        for _ in range (0,args.epoch):
            end=time.time()
            if self.it>args.epoch:
                break
            # start_epoch.update(time.time()-end)
            # self.it<args.epoch
            for (_,x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                                                                    self.loader_dict['train_ulb']):
                if self.it > args.num_train_iter:
                    break
                run_meter.update(time.time()-end)
                num_lb = x_lb.shape[0]
                num_ulb = x_ulb_w.shape[0]
                assert num_ulb == x_ulb_s.shape[0]
                x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
                y_lb = y_lb.cuda(args.gpu)

                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

                with amp_cm():
                    logits= self.model(inputs)
                    logits_x_lb = logits[:num_lb]
                    logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                    sup_loss = ce_loss(logits_x_lb, y_lb.long(),reduction='mean')

                    # hyper-params for update
                    time_p, p_model, label_hist = self.cal_time_p_and_p_model(logits_x_ulb_w, time_p, p_model, label_hist)
                    unsup_loss, mask,threshold_np = consistency_loss(args.dataset, logits_x_ulb_s,logits_x_ulb_w,
                                                        time_p,p_model,
                                                        'ce', use_hard_labels=args.hard_label)
                    
                    ent_loss = focal_loss(mask, logits_x_ulb_s, logits_x_ulb_w,p_model, label_hist)
                 
                    total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss

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
            start_epoch_meter.update(time.time()-end)
            sup_loss_meter.update(sup_loss.item())
            ent_loss_meter.update(ent_loss.item())
            unsup_loss_meter.update(unsup_loss.item())
            total_loss_meter.update(total_loss.item())
            time_p_meter.update(time_p.item())
            p_model_meter.update(p_model.mean().item())
            mask_ratio_meter.update(1.0-mask.float().mean().item())
            label_hist_meter.update(label_hist.mean().item())
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss_meter.avg
            tb_dict['train/total_loss'] = total_loss_meter.avg
            tb_dict['train/unsup_loss'] = unsup_loss_meter.avg
            tb_dict['train/ent_loss'] = ent_loss_meter.avg
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            if self.it % 50 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)
                # if tb_dict['eval/F1'] > best_eval_acc && tb_dict['eval/top-1-acc'] > best_eval_acc:
                if tb_dict['eval/F1'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/F1']
                    best_it = self.it
                self.print_fn(
                    f"{self.it} epoch, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} epoch")
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it== best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict,eval_dict['eval/best_acc']

    @torch.no_grad()
    def cal_time_p_and_p_model(self,logits_x_ulb_w, time_p, p_model, label_hist):
        prob_w = torch.softmax(logits_x_ulb_w, dim=1) 
        max_probs, max_idx = torch.max(prob_w, dim=-1)
        if time_p is None:
            time_p = max_probs.mean()
        else:
            time_p = time_p * 0.999 +  max_probs.mean() * 0.001
        if p_model is None:
            p_model = torch.mean(prob_w, dim=0)
        else:
            p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
        if label_hist is None:
            label_hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
            label_hist = label_hist / label_hist.sum()
        else:
            hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
            label_hist = label_hist * 0.999 + (hist / hist.sum()) * 0.001
        return time_p,p_model,label_hist


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
        features_1,labels=[],[]
        for _,x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y.long(), reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
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
