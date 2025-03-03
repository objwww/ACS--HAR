import torch
import torchvision
from torchvision import datasets
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
import numpy as np
import json
import os
import torch.utils.data as data
from sklearn import utils as skutils
import random
import math



def split_ssl_data(args, data, target, num_labels, num_classes, index=None, include_lb_to_ulb=False):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """  

    index_sup, index_unsup = [], []
    train_x_sup, train_y_sup, train_adj_sup, \
    train_x_unsup, train_y_unsup, train_adj_unsup = [], [], [], [], [], []
    for c in range(num_classes):
        train_x_c, train_y_c, indexc = skutils.shuffle(data[target == c],
                                                       target[target == c],
                                                       np.where(target == c)[0])
        train_x_sup.append(train_x_c[:math.ceil(num_labels * len(train_y_c))])
        train_y_sup.append(train_y_c[:math.ceil(num_labels * len(train_y_c))])
        index_sup.append(indexc[:math.ceil(num_labels * len(train_y_c))])
        train_x_unsup.append(train_x_c[math.ceil(num_labels * len(train_y_c)):])
        train_y_unsup.append(train_y_c[math.ceil(num_labels * len(train_y_c)):])
        index_unsup.append(indexc[math.ceil(num_labels * len(train_y_c)):])

    train_x_sup, train_y_sup = skutils.shuffle(np.concatenate(train_x_sup, 0),
                                                          np.concatenate(train_y_sup, 0))
    train_x_unsup, train_y_unsup = skutils.shuffle(np.concatenate(train_x_unsup, 0),
                                                                np.concatenate(train_y_unsup, 0))
    dump_path = os.path.join(args.save_dir, args.save_name, 'sampled_label_idx.npy')
    print(train_y_sup)
    return train_x_sup,train_y_sup,train_x_unsup,train_y_unsup


def sample_labeled_data(args, data, target,
                        num_labels, num_classes,
                        index=None, name=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    
    '''
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index
    print(data)
    dump_path = os.path.join(args.save_dir, args.save_name, 'sampled_label_idx.npy')

    if os.path.exists(dump_path):
        lb_idx = np.load(dump_path)
        lb_data = data[lb_idx]
        lbs = target[lb_idx]
        return lb_data, lbs, lb_idx
    samples_per_class = int(num_labels / num_classes)
    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        samples_per_class = int(num_labels / num_classes)
        idx = np.where(target == (c))[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])

    np.save(dump_path, np.array(lb_idx))







    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


def get_sampler_by_name(name):
    '''
    get sampler in torch.utils.data.sampler by name
    '''
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__
                               if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)


def get_data_loader(dset,
                    batch_size=None,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    data_sampler=None,
                    replacement=True,
                    num_epochs=None,
                    num_iters=None,
                    generator=None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)

        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
        else:
            num_replicas = 1

        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset) * num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas
        else:
            num_samples = len(dset)

        if data_sampler.__name__ == 'RandomSampler':
            print("——————————随机采样-------")
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler,
                          num_workers=num_workers, pin_memory=pin_memory)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot
def get_mean_and_std(train_data,channls,time_steps):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    num_channels =channls
    time_steps =time_steps
    print(num_channels,time_steps)
    mean = torch.zeros(num_channels)
    std = torch.zeros(num_channels)
    total_samples = 0

    for X in train_loader:
        batch_size = X.size(0)
        total_samples += batch_size * time_steps
        for channel in range(num_channels):
            channel_data = X[:, :, channel].contiguous().view(batch_size * time_steps, -1)
            mean[channel] += channel_data.mean()
            std[channel] += channel_data.std()

    mean.div_(total_samples)
    std.div_(total_samples)
    return list(mean.numpy()), list(std.numpy())
def ratio_split(ratio, train_x, train_y, seed, act_num):
    sup_cnt = 0
    index_sup, index_unsup = [], []
    train_x_sup, train_y_sup, train_adj_sup, \
    train_x_unsup, train_y_unsup, train_adj_unsup = [], [], [], [], [], []
    for c in range(act_num):
        train_x_c, train_y_c, indexc = skutils.shuffle(train_x[train_y == c],
                                                       train_y[train_y == c],
                                                       np.where(train_y == c)[0],
                                                       random_state=seed)

        train_x_sup.append(train_x_c[:math.ceil(ratio * len(train_y_c))])
        train_y_sup.append(train_y_c[:math.ceil(ratio * len(train_y_c))])
        index_sup.append(indexc[:math.ceil(ratio * len(train_y_c))])
        train_x_unsup.append(train_x_c[math.ceil(ratio * len(train_y_c)):])
        train_y_unsup.append(train_y_c[math.ceil(ratio * len(train_y_c)):])
        index_unsup.append(indexc[math.ceil(ratio * len(train_y_c)):])
        sup_cnt += math.ceil(ratio * len(train_y_c))

    train_x_sup, train_y_sup, index_sup = skutils.shuffle(np.concatenate(train_x_sup, 0),
                                                          np.concatenate(train_y_sup, 0),
                                                          np.concatenate(index_sup, 0),
                                                          random_state=seed)
    train_x_unsup, train_y_unsup, index_unsup = skutils.shuffle(np.concatenate(train_x_unsup, 0),
                                                                np.concatenate(train_y_unsup, 0),
                                                                np.concatenate(index_unsup, 0),
                                                                random_state=seed)
   
    return  train_x_sup, train_y_sup, index_sup,train_x_unsup, train_y_unsup, index_unsup