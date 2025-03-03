import torch
import torch.nn.functional as F
from train_utils import ce_loss,mse_loss
import numpy as np
from sklearn.metrics import f1_score


class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value



def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val



def entropy_loss(mask, logits_s, logits_w, prob_model, label_hist):
    logits_s = logits_s[mask]
    prob_s = logits_s.softmax(dim=-1) 
    _, pred_label_s = torch.max(prob_s, dim=-1)#
    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_w.dtype)

    hist_s = hist_s / hist_s.sum()

    prob_model = prob_model.reshape(1, -1) #
    label_hist = label_hist.reshape(1, -1)
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)
    loss = -(mod_prob_model * torch.log(mod_mean_prob_s + 1e-12))
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()
def focal_loss(mask, logits_s, logits_w, prob_model, label_hist):
    gama = 0.25 
    alpha = 2.0
    logits_s = logits_s[mask]
    logits_w = logits_w[mask]
    prob_s = F.log_softmax(logits_s, dim=-1)
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    ce_loss = F.cross_entropy(prob_s,max_idx)
    pt = torch.exp(-ce_loss)
    F_loss = alpha*(1-pt)**gama * ce_loss
    return F_loss.mean()



def consistency_loss(dataset, logits_s, logits_w,time_p,p_model, name='ce', use_hard_labels=True):
    assert name in ['ce', 'L2','kl']
    logits_w = logits_w.detach() 
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1) 
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        p_cutoff = time_p
        p_model_cutoff = p_model / torch.max(p_model,dim=-1)[0]
        threshold = p_cutoff * p_model_cutoff[max_idx]
        threshold_np = threshold.mean().cpu().detach().numpy()
        mask = max_probs.ge(threshold)

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask.float()
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask.float()
        return masked_loss.mean(), mask,threshold_np
  
    else:
        assert Exception('Not Implemented consistency_loss')
def build_mixup_fn(x, y, gpu, alpha=1.0, is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).cuda(gpu)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lams