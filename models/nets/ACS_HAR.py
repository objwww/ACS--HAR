import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os
import math
import torch.nn.functional as F

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv1d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
   
    def forward_slicing(self, x: Tensor) -> Tensor:

        x = x.clone() 
        
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1) 
        x1=self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):
    

    def __init__(self,
                 dim,
                 in_chann,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div
        self.in_chann=in_chann

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv1d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv1d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = self.mlp(x)
        x = self.drop_path(x)+shortcut
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 in_chann,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                in_chann=in_chann,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks_list)
    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x

class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj2 = nn.Conv1d(in_chans, embed_dim, kernel_size=4,stride=4,bias=False)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
       
        x2 = self.proj2(x)
        x = self.norm(x2)
        return x


class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv1d(dim, 2 * dim, kernel_size=2, stride=1, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x
class GRU_attention(nn.Module):
    def __init__(self,in_chans,hidden_num):
        super().__init__()
        self.lstm1 = nn.GRU(in_chans, hidden_num,bidirectional=True,num_layers=2, batch_first=True)
        self.tanh=nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        self.lstm1.flatten_parameters()
        x,_=self.lstm1(x)
        x = self.tanh(x)
        return x



class ACS_HAR(nn.Module):

    def __init__(self,
                 in_chans,
                 num_classes,
                 embed_dim,
                 depths,
                 mlp_ratio,
                 n_div,
                 patch_size,
                 patch_stride,
                 patch_size2, 
                 patch_stride2,
                 patch_norm,
                 feature_dim,
                 drop_path_rate,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 fork_feat,
                 init_cfg,
                 pretrained,
                 hidden_num,
                 pconv_fw_type,
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm1d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.feature_dim=feature_dim
        self.batch_norm=norm_layer(in_chans)
        self.num_classes=num_classes
        self.tanh = nn.Tanh()

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

  
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               depth=depths[i_stage],
                               in_chann=in_chans,
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
            stages_list.append(stage)
          

            if i_stage < self.num_stages - 1:
                stages_list.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** i_stage),
                                 norm_layer=norm_layer)
                )
        self.stages = nn.Sequential(*stages_list)
        self.fork_feat = fork_feat
        if self.fork_feat:
            self.forward = self.forward_det
    
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls
            print(self.num_features)
            self.GRU_attention=GRU_attention(self.num_features,hidden_num)
            self.fc= nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(128,hidden_num, 1, bias=False),
                norm_layer(hidden_num),
                act_layer()
            )
            self.head = nn.Linear(hidden_num, num_classes) \
                if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.trunc_normal_(param, std=0.02)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection by loading imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def forward_cls(self, x):
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = self.stages(x)
        x = x.permute(0, 2, 1)
        x = self.GRU_attention(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
class build_ACS_HAR:
    def __init__(self,
                 in_chans, 
                 embed_dim,
                 depths,
                 mlp_ratio,
                 n_div,
                 feature_dim,
                 drop_path_rate,
                 hidden_num,
                 act_layer):
        self.in_chans=in_chans
        self.embed_dim=embed_dim
        self.depth=depths
        self.mlp_ratio=mlp_ratio
        self.n_div=n_div
        self.patch_size=4
        self.patch_stride=4
        self.patch_size2=2
        self.patch_stride2=2
        self.patch_norm=True
        self.feature_dim=feature_dim
        self.drop_path_rate=drop_path_rate
        self.layer_scale_init_value=0
        self.norm_layer='BN'
        self.act_layer=act_layer
        self.fork_feat=False
        self.init_cfg=None
        self.pretrained=None
        self.pconv_fw_type='split_cat'
        self.hidden_num=hidden_num

    def build(self, num_classes):
        return ACS_HAR(in_chans=self.in_chans,num_classes=num_classes,embed_dim=self.embed_dim,depths=self.depth,mlp_ratio=self.mlp_ratio,n_div=self.n_div,patch_size=self.patch_size,
        patch_stride=self.patch_stride,patch_size2=self.patch_size2,patch_stride2=self.patch_stride2,patch_norm=self.patch_norm,feature_dim=self.feature_dim,drop_path_rate=self.drop_path_rate,
        layer_scale_init_value=self.layer_scale_init_value,norm_layer=self.norm_layer,act_layer=self.act_layer,fork_feat=self.fork_feat,init_cfg=self.init_cfg,pretrained=self.pretrained,pconv_fw_type=self.pconv_fw_type,hidden_num=self.hidden_num)

if __name__ == '__main__':
    wrn_builder = build_ACS_HAR(16,(1,2,8,2),2,0.1,'RELU')
    wrn = wrn_builder.build(4)
    print(wrn)