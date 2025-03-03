from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.transforms import *

import torchvision
from PIL import Image
import numpy as np
import copy
import cv2
import torch


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot
      
        self.transform = transform
        if self.is_ulb:
            self.strong_transform = strong_transform
           
        else:
            self.strong_transform = None

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.data[idx]
        img_w = self.transform(img)

        if not self.is_ulb:
            return idx, img_w, target
        else:
            if self.alg == 'ACS_HAR':
                return idx, img_w, self.strong_transform(img)
            elif self.alg == 'pimodel':
                return idx, img_w, self.transform(img)
            elif self.alg == 'pseudolabel':
                return idx, img_w
            elif self.alg == 'fullysupervised':
                return idx

    def __len__(self):
        return len(self.data)
