import torch
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from .data_utils import split_ssl_data, sample_labeled_data,get_mean_and_std
from .dataset import BasicDataset
from collections import Counter
import torchvision
import numpy as np
from torchvision import transforms
import json
import os

import random
from .augmentation.transforms import *

from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
from datasets.DistributedProxySampler import DistributedProxySampler

import gc
import sys
import copy
mean, std = {}, {}
mean['uci'] =[-4.971134e-06, -2.2835648e-06, -2.1507765e-06, 3.956752e-06, 3.956752e-06, 3.956752e-06, 0.006287107, 0.00022465202, 0.00067576545]
mean['wisdm']=[2.2292901e-10, 1.11374625e-10, -1.9386825e-10]
mean['mhealth']=[-1.1424381e-10, 2.9679834e-10, 7.896956e-10, -3.2984954e-10, -1.51362e-11, -8.378738e-11, -1.0994494e-09, 3.2020653e-11, 4.650724e-10, -2.1310297e-11, 1.341555e-11, -1.8035541e-10, 3.4026545e-10, 5.6385795e-11, -2.3894034e-10, -5.2322485e-10, -2.9265773e-10, -6.1464933e-10, -2.1843975e-11, 2.587416e-11, 4.968722e-12]
mean['pamap']=[-4.1622484e-06, -1.3927707e-05, 1.6058347e-06, 3.4209122e-06, 4.0792047e-06, 7.430613e-07, -3.75495e-07, 1.3335715e-06, -2.0814725e-06, -5.8404106e-07, 8.794538e-07, 4.2596486e-07, 1.024724e-06, -3.459557e-07, 6.2934305e-06, -1.2506863e-06, 5.629679e-07, 9.1309295e-07, 4.647155e-05, -2.4671314e-07, -1.1060338e-06, -3.0529878e-07, 1.0794533e-06, -1.0293697e-06, -3.3991762e-07, -1.6076262e-07, -1.9941476e-07, -3.2447576e-07, -1.4584239e-06, 1.1164667e-06, -1.6859627e-07, -2.3709244e-06, 7.715573e-07, 3.146002e-07, -6.9495975e-08, -4.2615313e-05, -1.9929384e-07, 7.354335e-07, 6.8589916e-06, -3.2297585e-06, 2.266678e-06, 1.956829e-06, -1.7522426e-07, 4.685327e-07, -2.0127939e-07, 2.7033202e-06, 7.532038e-09, 1.4618856e-06, -3.4557913e-06, -1.613245e-06, -2.0793395e-06, -1.7163521e-06]
mean['usc']=[-5.1983067e-11, -3.647787e-10, 1.2056216e-10, 6.5685504e-12, -1.0828173e-10, 4.632799e-12]
std['usc']=[0.0044852896, 0.004176442, 0.0047542676, 0.0065062866, 0.0065997047, 0.006313341]
std['uci']= [0.001012599, 0.0006662502, 0.00057895255, 0.0019121259, 0.0019121259, 0.0019121259, 0.0009995692, 0.0006490474, 0.00055597187]
std['wisdm']=[0.005134031, 0.0058212136, 0.006246732]
std['mhealth']=[0.004373709, 0.004512783, 0.0030758448, 0.0045759506, 0.003992614, 0.0043515265, 0.0012645212, 0.0020616804, 0.002542372, 0.0049177706, 0.0046820147, 0.004047421, 0.0041067256, 0.004240015, 0.003616083, 0.0017318444, 0.0019810298, 0.001817396, 0.005163039, 0.0057032723, 0.0048265266]
std['pamap']=[0.00020886053, 0.002650257, 0.0037133466, 0.004479961, 0.0039421977, 0.0036762473, 0.0044387667, 0.0038699077, 0.0059821205, 0.0056785857, 0.0056871753, 0.0028380607, 0.0032777837, 0.0029076212, 0.0027630704, 0.0024502482, 0.002434027, 0.0026344326, 0.0015699579, 0.005100235, 0.0038777555, 0.0024248278, 0.0050508548, 0.003904671, 0.002406991, 0.005633518, 0.005345563, 0.0060614, 0.0018811651, 0.0016540285, 0.0016812192, 0.0016442054, 0.0016595672, 0.0014218707, 0.001392698, 0.0036673928, 0.0044298815, 0.004718375, 0.0050001075, 0.0044356156, 0.004752579, 0.005058902, 0.0056571467, 0.005590571, 0.00535998, 0.0028398703, 0.0032988223, 0.0022984692, 0.0029476043, 0.0018489076, 0.0021731837, 0.0020114318]
mean['hapt']=[0.006138577, 0.0005118975, 0.0009246955, -7.779412e-06, 1.310863e-06, -3.4981313e-05]
std['hapt']=[0.0008636861, 0.000588037, 0.00047996812, 0.0016483896, 0.0017091297, 0.0011395776]

def get_transform(mean,std,train=True):#图像变换函数

    if train:
        weak_transform = Compose([
                        Scaling(0.7),
                        ToTensor(),
                        Normalize(mean,std)
                        ])
        
        strong_transform = Compose([
                        MagnitudeWrap(sigma=0.2,knot=4,p=1.0),
                        TimeWarp(sigma=0.2,knot=8,p=1.0),
                        Scaling(0.7),
                        ToTensor(),
                        Normalize(mean,std)
                        ])
        return weak_transform,strong_transform
    else:
        transform =  Compose([
                        ToTensor(),
                        Normalize(mean,std)
                        ])
        return transform,None


class SSL_Dataset:
    """
    SSL_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self,
                 args,
                 alg='fixmatch',
                 name='uci',
                 train=True,
                 num_classes=10,
                 inchans=9,
                 data_dir='./data'):
        """
        Args
            alg: SSL algorithms
            name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        self.args = args
        self.alg = alg
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.inchans=inchans
       
        self.weak_transform,self.strong_transform= get_transform(mean[name], std[name],train)

    def get_data(self, svhn_extra=True):
        # """
        # get_data returns data (images) and targets (labels)
        # shape of data: B,T, C
        # shape of labels: B,
        # """
        if self.name == 'pamap':
            self._path = self.data_dir+'/'+'data/PAMAP2'+'/'
            self._channel_num = 52
            self._length = 128
            self._act_num = 12
        elif self.name == 'uci':
            self._path = self.data_dir+'/'+'data/UCI_HAR' +'/'
            self._channel_num = 9
            self._length = 128
            self._act_num = 6
        elif self.name == 'wisdm':
            self._path = self.data_dir+'/'+'data/WISDM_180' +'/'
            self._channel_num = 3
            self._length = 180
            self._act_num = 6
        elif self.name == 'mhealth':
            self._path = self.data_dir+'/'+'data/Mhealth' + '/'
            self._channel_num = 21
            self._length = 128
            self._act_num = 12
        elif self.name == 'hapt':
            self._path = self.data_dir+'/'+'data/HAPT' + '/'
            self._channel_num = 6
            self._length = 128
            self._act_num = 6
        elif self.name == 'usc':
            self._path = self.data_dir+'/'+'data/USC_HAD' + '/'
            self._channel_num = 6
            self._length = 100
            self._act_num = 12
        if self.train:
            data = np.load(self._path + 'x_train.npy')
            target = np.load(self._path + 'y_train.npy')
        else:
            data = np.load(self._path + 'x_test.npy')
            target = np.load(self._path + 'y_test.npy')
        return data, target
    

    def get_dset(self, is_ulb=False,
                 strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.
        
        Args
            is_ulb: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True。
            onehot: If True, the label is not integer, but one-hot vector.
        """
        data, targets = self.get_data()
        num_classes =  self.num_classes
        transform = self.weak_transform

        return BasicDataset(self.alg, data, targets, num_classes,  transform,
                            is_ulb, strong_transform, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=False,
                     strong_transform=None, onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        data, targets = self.get_data()
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(self.args, data, targets,
                                                                        num_labels,  self.num_classes,
                                                                       index, include_lb_to_ulb)
        mean,std=get_mean_and_std(data,data.shape[2],data.shape[1])
        print(mean)
        print(std)
        print("data",data.shape)
        print("lb_data",lb_data.shape)
        print("lb_targets",lb_targets.shape)
        print("ulb_data",ulb_data.shape)
        print("ulb_targets",ulb_targets.shape)
        # output the distribution of labeled data for remixmatc
        count = [0 for _ in range(self._act_num)]
        for c in lb_targets:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(self.name) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)
        # print(Counter(ulb_targets.tolist()))
        lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
                                self.weak_transform, False, None, onehot)

        ulb_dset = BasicDataset(self.alg, ulb_data, ulb_targets, self.num_classes,
                                 self.weak_transform, True, self.strong_transform, onehot)
        print(lb_data.shape)
        print(ulb_data.shape)
        return lb_dset, ulb_dset
