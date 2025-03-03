import random

import torch
import scipy.interpolate

from .augmentation import *



class noise_transform_vectorized:
    def __init__(self,mix,max):
        self.mix=mix
        self.max=max

  

    def __call__(self, data):
        # print('### Scaling')
        return data
    def forward(self, data):
        sigma= self.mix+float(self.max -  self.mix)*random.random()
        return noise_transform_vectorized(data,sigma)
    
class scaling_transform_vectorized:
    def __init__(self,mix,max):
        self.mix=mix
        self.max=max
  

    def __call__(self, data):
        # print('### Scaling')

        return data

    def forward(self, data):
        sigma= self.mix+float(self.max -  self.mix)*random.random()
        return scaling_transform_vectorized(data, sigma=sigma)

class rotation_transform_vectorized:
    def __init__(self):
         pass
    
  


    def __call__(self, data):

        return data

    def forward(self, data):
        return rotation_transform_vectorized(data)
    
class negate_transform_vectorized:
    def __init__(self):
        pass
    
  

    def __call__(self, data):
        return data

    def forward(self, data):
        return negate_transform_vectorized(data)
    
class time_flip_transform_vectorized:
    def __init__(self):
       pass
  

    def __call__(self, data):

        return data

    def forward(self, data):
        return time_flip_transform_vectorized(data)

class time_segment_permutation_transform_improved:
    def __init__(self,mix,max):
        self.mix=mix
        self.max=max
    
  
    
    def __call__(self, data):

        return data

    def forward(self, data):
        sigma= self.mix+float(self.max -  self.mix)*random.random()
        return time_segment_permutation_transform_improved(data,sigma)
    
class time_warp_transform_low_cost:
    def __init__(self,mix,max,num_knots,num_splines):
        self.mix=mix
        self.max=max
        self.num_knots=num_knots
        self.num_splines=num_splines

        
    
  

    def __call__(self, data):

        return data

    def forward(self, data):
        sigma= self.mix+float(self.max -  self.mix)*random.random()
        return time_warp_transform_low_cost(data,sigma,self.num_knots,self.num_splines)
 
class channel_shuffle_transform_vectorized:
    def __init__(self):
        pass
    
  

    def __call__(self, data):

        return data

    def forward(self, data):
        return channel_shuffle_transform_vectorized(data)


class Raw:
    def __init__(self,p):
        self.p=p
  


    def __call__(self, data):
        return data


class CutPiece2C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return cut_piece2C(data, self.sigma)


class CutPiece3C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return cut_piece3C(data, self.sigma)


class CutPiece4C:
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return cut_piece4C(data, self.sigma)


class CutPiece5C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return cut_piece5C(data, self.sigma)


class CutPiece6C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return cut_piece6C(data, self.sigma)


class CutPiece7C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return cut_piece7C(data, self.sigma)


class CutPiece8C:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return cut_piece8C(data, self.sigma)


# class Jitter:
#     def __init__(self, sigma, p):
#         self.sigma = sigma
#         self.p = p

#     def __call__(self, data):
#         # print('### Jitter')

#         if random.random() < self.p:
#             yield from self.forward(data)
#         return data

#     def forward(self, data):
#         return jitter(data, sigma=self.sigma)
class Jitter:
    def __init__(self,max):
        # self.mix=mix
        self.max=max

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        # sigma= self.mix+float(self.max -  self.mix)*random.random()
        return jitter(data, sigma=self.max)

class SlideWindow:
    def __init__(self, horizon, stride):
        self.horizon = horizon
        self.stride = stride
        

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return slidewindow(data, self.horizon, self.stride)


class Scaling:
    def __init__(self,max):
        # self.mix=mix
        self.max=max

    def __call__(self, data):
        # print('### Scaling')
        return self.forward(data)

    def forward(self, data):
        # sigma= self.mix+float(self.max -  self.mix)*random.random()
        return scaling_s(data, sigma=self.max)

class Rotation:
    def __init__(self):
        pass
  

    def __call__(self, data):
        # print('### Scaling')
        return data

    def forward(self, data):
        return rotation_s(data)
    
class Permutation:
    def __init__(self,max,seg_mode="equal"):
        # self.mix=mix
        self.max=max
        self.seg_mode=seg_mode

    def __call__(self, data):
        # print('### Scaling')
        return data

    def forward(self, data):
        # sigma= self.mix+float(self.max -  self.mix)*random.random()
        return permutation(data,self.max,self.seg_mode)
    
class Rotation2D:
    def __init__(self,mix,max):
        self.mix=mix
        self.max=max
        # 
    
  
    def __call__(self, data):
        # print('### Scaling')
        return data

    def forward(self, data):
        sigma= self.mix+float(self.max -  self.mix)*random.random()
        return rotation2d(data,sigma)



class CutPF:
    def __init__(self,sigma):
        self.sigma = sigma
        # 
  

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return cutPF(data, self.sigma)


class Cutout:
    def __init__(self,mix,max, p=0.4):
        self.mix=mix
        self.max=max
        self.p = p
        # 
    def __call__(self, data):
    
        return self.forward(data)

    def forward(self, data):
        return cutout(data, self.sigma)


class Crop:
    def __init__(self,length=112):
        self.length = length
        # 
    def __call__(self, data):
        # print('### Cutout')
        return self.forward(data)

    def forward(self, data):
        if isinstance(data, torch.Tensor):  # Check if data is a tensor
            # Handle the tensor data appropriately
            # For example, you can convert it to a numpy array to get its length
            data = data.numpy()
        start = np.random.randint(data.shape[0] - self.length + 1)
        return data[start: start+self.length]
class Random_Filp:
    def __init__(self):
        pass
        # 
    def __call__(self, data):
        # print('### Cutout')
        return self.forward(data)

    def forward(self, data):
        if isinstance(data, torch.Tensor):  # Check if data is a tensor
            data = data.numpy()
        flip_prob = np.random.choice([True,False])
        if flip_prob:
            flipped_seq = np.flip(data)
            return flipped_seq
        else:
            return data

class MagnitudeWrap:
    def __init__(self, sigma, knot, p):
        self.sigma = sigma
        self.knot = knot
        self.p = p

    def __call__(self, data):
        # print('### MagnitudeWrap')

        if random.random() < self.p:
            return self.forward(data)

        return data

    def forward(self, data):
        return magnitude_warp_s(data, sigma=self.sigma, knot=self.knot)


class TimeWarp:
    def __init__(self, sigma, knot, p):
        self.sigma = sigma
        self.knot = knot
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            return self.forward(data)

        return data

    def forward(self, data):
        return time_warp_s(data, sigma=self.sigma, knot=self.knot)


class WindowSlice:
    def __init__(self,reduce_ratio):
        self.reduce_ratio = reduce_ratio
        # self.p = p
        #

    def forward(self, data):
        return window_slice_s(data, reduce_ratio=self.reduce_ratio)


class WindowWarp:
    def __init__(self,window_ratio, scales):
        self.window_ratio = window_ratio
        self.scales = scales
    def __iter__(self):
        return self

    def __call__(self, data):
        return self.forward(data)

       

    def forward(self, data):
        return window_warp_s(data, window_ratio=self.window_ratio, scales=self.scales)


class ToTensor:
    '''
    Attributes
    ----------
    basic : convert numpy to PyTorch tensor

    Methods
    -------
    forward(img=input_image)
        Convert HWC OpenCV image into CHW PyTorch Tensor
    '''

    def __init__(self, basic=False):
        self.basic = basic

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        '''
        Parameters
        ----------
        img : opencv/numpy image
        Returns
        -------
        Torch tensor
            BGR -> RGB, [0, 255] -> [0, 1]
        '''
        ret = torch.from_numpy(img.copy()).type(torch.FloatTensor)
        return ret
class Normalize:
    def __init__(self, mean,std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        if not isinstance(img,torch.Tensor):
            img=torch.tensor(img)
        img=(img-self.mean)/self.std

        return img

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        
        for t in self.transforms:
            img = t(img)
        return img
    
    def insert(self, index, transform):

        self.transforms.insert(index, transform)


def augment_list():  
    l = [
        noise_transform_vectorized(0.03,0.07),
        Permutation(8,seg_mode='equal'),
        # Jitter(0.05),
        Rotation(),
        time_flip_transform_vectorized(),
        MagnitudeWrap(0.1,0.3,knot=4),
        negate_transform_vectorized(),
        time_segment_permutation_transform_improved(4,7),
        time_warp_transform_low_cost(0.2,0.5,num_knots=4,num_splines=150)


        # WindowWarp(window_ratio=0.3,scales=[0.5,2.2]),
        # WindowSlice(reduce_ratio=0.5),
        # rotation_transform_vectorized(),
    ]
    return l

    
class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        crop=Random_Filp()
        for op in ops:
            img = op(img) 
        img=crop(img)
        return img
