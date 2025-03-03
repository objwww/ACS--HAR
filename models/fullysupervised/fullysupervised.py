from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os
import contextlib
from .fullysupervised_utils import build_mixup_fn,one_hot,plot_tsne,plot_embedding
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller,AverageMeter,EarlyStopping
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from copy import deepcopy
import time


class FullySupervised:
    def __init__(self, net_builder, num_classes,
                 num_eval_iter=1000, tb_log=None, ema_m=0.999, logger=None):
        """
        class FullySupervised contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            it: initial iteration count
            num_eval_iter: frequency of evaluation.
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(FullySupervised, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.num_eval_iter = num_eval_iter
        self.tb_log = tb_log
        # self.mixup_fn=build_mixup_fn(self,num_classes)
        self.optimizer = None
        self.scheduler = None

        self.it = 0

        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.ema_m = ema_m
        self.ema_model = deepcopy(self.model)
        self.tsne = TSNE(n_components=2, random_state=112)
        self.bn_controller = Bn_Controller()
        self.earlyStopping=EarlyStopping(patience=10,verbose=True)
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args):

        ngpus_per_node = torch.cuda.device_count()

        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        best_eval_acc, best_it = 0.0, 0
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)
        sup_loss_meter=AverageMeter()
        total_loss_meter=AverageMeter()
        lr_meter=AverageMeter()
        start_epoch_meter=AverageMeter()
        run_meter=AverageMeter()
        for _  in range (0,args.epoch):
            end=time.time()
            if self.it>args.epoch:
                break
            for _,x_lb, y_lb in self.loader_dict['train_lb']:
                x_lb = x_lb.cuda(args.gpu)
                y_lb = y_lb.cuda(args.gpu)
                run_meter.update(time.time()-end)
                num_lb = x_lb.shape[0]
                with amp_cm():
                    logits_x_lb= self.model(x_lb)
                    sup_loss = ce_loss(logits_x_lb, y_lb.long(), reduction='mean')
                    total_loss = sup_loss
                    sup_loss_meter.update(sup_loss.item())
                    total_loss_meter.update(total_loss.item())
                    lr_meter.update(self.optimizer.param_groups[0]['lr'])
                # parameter updates
                if args.amp:
                    scaler.scale(total_loss).backward()
                    if (args.clip > 0):
                        scaler.unscale_(self.optimizer)
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

                # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss_meter.avg
            tb_dict['train/total_loss'] = total_loss_meter.avg
            tb_dict['lr'] = lr_meter.avg
            tb_dict['train/prefecth_time'] = start_epoch_meter.avg/1000.
            tb_dict['train/run_time'] =run_meter.avg/1000.

            if self.it % 50 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                self.earlyStopping(eval_dict['eval/loss'],self.model)
                if self.earlyStopping.early_stop:
                    print("Early stoping")
                tb_dict.update(eval_dict)
                if self.it % self.num_eval_iter == 0:
                        eval_dict = self.evaluate(args=args)
                save_path = os.path.join(args.save_dir, args.save_name)
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
        # return eval_dict

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
        labels=[]

        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits= self.model(x)
            _, predicted = torch.max(logits, 1)
            loss = F.cross_entropy(logits, y.long(), reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
            labels.append(y.cpu().numpy())
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        F1=f1_score(y_true, y_pred,average='macro')
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num,'eval/F1':F1, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5}

    def save_model(self, save_name, save_path,):
        if self.it < 20:
            return
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
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.print_fn('model loaded')

    # Abandoned in PiModel
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
