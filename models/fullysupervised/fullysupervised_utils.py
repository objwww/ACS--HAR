import torch
import torch.nn.functional as F
from train_utils import ce_loss
import numpy as np
from timm.data import Mixup
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value

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
    return mixed_x, mixed_y, lam
def one_hot(targets, nClass, gpu):
    logits = torch.zeros(targets.size(0), nClass).cuda(gpu)
    return logits.scatter_(1, targets.unsqueeze(1), 1)


def plot_tsne(tsne_results, labels,epoch):
    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title(f't-SNE Visualization at Epoch {epoch + 1}')
    plt.savefig(f'tsne_epoch_{epoch + 1}.png')
    plt.close()

def plot_embedding(data, label,epoch):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    scaler = StandardScaler()
    plt.figure(figsize=(10, 8))
    scatter = plt.StandardScaler(data[:, 0], data[:, 1], c=label,s=3.0, cmap='viridis', alpha=0.7,marker = 'o')
    plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(10)])
    plt.xticks([])
    plt.yticks([])
    plt.title(f't-SNE Visualization at Epoch {epoch + 1}')
    plt.savefig(f'1tsne_epoch_{epoch + 1}.png')
    plt.close()
    return fig