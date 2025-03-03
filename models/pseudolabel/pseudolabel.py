import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
from train_utils import AverageMeter
import os
import contextlib
from train_utils import EMA, Bn_Controller
from collections import Counter
from .pseudolabel_utils import consistency_loss
from train_utils import ce_loss, wd_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import *
from copy import deepcopy
from sklearn.preprocessing import StandardScaler


class PseudoLabel:
    def __init__(self, net_builder, num_classes, lambda_u,
                 num_eval_iter=1000, tb_log=None, ema_m=0.999, logger=None):
        """
        class PseudoLabel contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            lambda_u: ratio of unsupervised loss to supervised loss
            it: initial iteration count
            num_eval_iter: frequency of evaluation.
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(PseudoLabel, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes

        self.ema_m = ema_m

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = deepcopy(self.model)

        self.num_eval_iter = num_eval_iter
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

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args):

        ngpus_per_node = torch.cuda.device_count()

        # EMA init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label = selected_label.cuda(args.gpu)
        classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)
        sup_loss_meter=AverageMeter()
        total_loss_meter=AverageMeter()
        start_epoch=AverageMeter() 
        end_epoch=AverageMeter()
        unsup_loss_meter=AverageMeter()
        end=time.time()
        for _ in range (0,args.epoch):
            sup_loss_meter.reset()
            total_loss_meter.reset()
            unsup_loss_meter.reset()
            start_epoch.update(time.time()-end)
     
            for (_,x_lb, y_lb), (x_ulb_idx, x_ulb_w) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):

                # prevent the training iterations exceed args.num_train_iter
                if self.it > args.epoch:
                    break
                unsup_warmup = np.clip(self.it / (args.unsup_warmup_pos * args.epoch),
                                    a_min=0.0, a_max=1.0)

                x_lb, x_ulb_w = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu)
                x_ulb_idx = x_ulb_idx.cuda(args.gpu)
                y_lb = y_lb.cuda(args.gpu)

                num_lb = x_lb.shape[0]
                if args.use_flex:
                    pseudo_counter = Counter(selected_label.tolist())
                    if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                        for i in range(args.num_classes):
                            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

                # inference and calculate sup/unsup losses
                with amp_cm():
                    logits_x_lb= self.model(x_lb)
                    self.bn_controller.freeze_bn(self.model)
                    logits_x_ulb_w = self.model(x_ulb_w)
                    self.bn_controller.unfreeze_bn(self.model)

                    sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                    unsup_loss, select, pseudo_lb = consistency_loss(logits_x_ulb_w, classwise_acc, self.it, args.dataset,
                                                                    args.p_cutoff, use_flex=False)
                    if x_ulb_idx[select == 1].nelement() != 0:
                        selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

                    total_loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup

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

            end_epoch.update(time.time()-end)
            sup_loss_meter.update(sup_loss.item())
            unsup_loss_meter.update(unsup_loss.item())
            total_loss_meter.update(total_loss.item())

            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss_meter.avg
            tb_dict['train/total_loss'] = total_loss_meter.avg
            tb_dict['train/unsup_loss'] = unsup_loss_meter.avg
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_epoch.avg / 1000.
            tb_dict['train/run_time'] = end_epoch.avg / 1000.
            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 40 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if args.rank % ngpus_per_node == 0:
                    self.save_model('latest_model.pth', save_path)

            if self.it % 1 == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/F1'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/F1']
                    best_it = self.it
                self.print_fn(
                        f"{self.it} epoch, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} epoch")

                if not args.rank % ngpus_per_node == 0:

                    if self.it== best_it:
                        self.save_model('model_best.pth', save_path)

                    if  self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)
            del tb_dict
            self.it += 1
        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict,eval_dict['eval/best_acc']

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

        for _,x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits= self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
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
        F1=f1_score(y_true, y_pred,average='macro')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num,'evel/F1':F1, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

    def save_model(self, save_name, save_path):
        if self.it < 100:
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
                    'it': self.it + 1,
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

    # Abandoned in Pseudo Label
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
