import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.fusion import Fusion
from lib.models.DID import DID

# from lib.losses.centernet_loss import compute_centernet3d_loss
from lib.losses.head_distill_loss import compute_head_distill_loss
from lib.losses.feature_distill_loss import compute_backbone_l1_loss
from lib.losses.feature_distill_loss import compute_backbone_resize_affinity_loss
from lib.losses.feature_distill_loss import compute_backbone_local_affinity_loss
from lib.losses.myloss import spearman_loss

class DID_Distill(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', downsample=4, flag='training', mean_size=None, model_type='distill', cfg=None):
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.cfg = cfg
        self.centernet_rgb = DID(backbone=backbone, neck=neck, mean_size=mean_size, model_type=model_type)
        self.centernet_depth = DID(backbone=backbone, neck=neck, mean_size=mean_size, model_type=model_type)

        for i in self.centernet_depth.parameters():
            i.requires_grad = False

        channels = self.centernet_rgb.backbone.channels         # [16, 32, 64,  128, 256, 512 ]
        tea_channels = self.centernet_depth.backbone.channels   # [16, 32, 128, 256, 512, 1024]
        input_channels = channels[2:]
        out_channels = channels[2:]
        mid_channel = channels[-1]
        rgb_fs = nn.ModuleList()
        for idx, in_channel in enumerate(input_channels):
            rgb_fs.append(Fusion(in_channel, mid_channel, out_channels[idx], idx < len(input_channels)-1))
        self.rgb_fs = rgb_fs[::-1]

        if 'dlaup_kd' in self.cfg['kd_type']:
            self.adapt_list = ['adapt_layer4', 'adapt_layer8','adapt_layer16','adapt_layer32']
            for i, adapt_name in enumerate(self.adapt_list):
                fc = nn.Sequential(
                    nn.Conv2d(channels[i+2], tea_channels[i+2], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(tea_channels[i+2], tea_channels[i+2], kernel_size=1, padding=0, bias=True)
                )
                #self.fill_fc_weights(fc)
                self.__setattr__(adapt_name, fc)
        else:
            self.adapt_list = ['adapt_layer8','adapt_layer16','adapt_layer32']
            for i, adapt_name in enumerate(self.adapt_list):
                fc = nn.Sequential(
                    nn.Conv2d(channels[i+3], tea_channels[i+3], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(tea_channels[i+3], tea_channels[i+3], kernel_size=1, padding=0, bias=True)
                )
                #self.fill_fc_weights(fc)
                self.__setattr__(adapt_name, fc)

        self.flag = flag
        
    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    # def forward(self, input, target=None):
    def forward(self, input, coord_ranges, calibs, target=None, K=50, mode='train'):
    
        if mode == 'train' and target != None:
            rgb = input['rgb']
            depth = input['depth']

            rgb_feat, rgb_outputs, rgb_fs_feat = self.centernet_rgb(rgb, coord_ranges, calibs, target, K, mode)
            depth_feat, depth_outputs, depth_fs_feat = self.centernet_depth(depth, coord_ranges, calibs, target, K, mode)

            if 'dlaup_kd' not in self.cfg['kd_type']:
                ### rgb feature fusion
                ### References: Distilling Knowledge via Knowledge Review, CVPR'21
                shapes = [rgb_feat_item.shape[2:] for rgb_feat_item in rgb_feat[::-1]]
                out_shapes = shapes
                x = rgb_feat[::-1]

                results = []
                out_features, res_features = self.rgb_fs[0](x[0], out_shape=out_shapes[0])
                results.append(out_features)
                for features, rgb_f, shape, out_shape in zip(x[1:], self.rgb_fs[1:], shapes[1:], out_shapes[1:]):
                    out_features, res_features = rgb_f(features, res_features, shape, out_shape)
                    results.insert(0, out_features)

                ### adapt layer
                distill_feature = []
                for i, adapt in enumerate(self.adapt_list):
                    distill_feature.append(self.__getattr__(adapt)(results[i+1]))
                
                # -------------------------------------------------------------------------------------
                head_loss, backbone_loss_l1, backbone_loss_affinity = None, None, None
                if 'head_kd' in self.cfg['kd_type']:
                    head_loss, _ = compute_head_distill_loss(rgb_outputs, depth_outputs, target)

                if 'fg_kd' in self.cfg['kd_type']:
                    fg_backbone_loss_l1 = compute_backbone_l1_loss(distill_feature, depth_feat[-3:], target)
                    backbone_loss_l1 = fg_backbone_loss_l1
                
                if 'affinity_kd' in self.cfg['kd_type']:
                    backbone_loss_affinity = compute_backbone_resize_affinity_loss(distill_feature, depth_feat[-3:])

                if 'spearman_kd' in self.cfg['kd_type']:
                    relation_loss = spearman_loss(distill_feature, depth_feat[-3:])

                return rgb_outputs, backbone_loss_l1, backbone_loss_affinity, head_loss
        
            else:
                rgb_fs_feat = rgb_fs_feat[::-1]
                depth_fs_feat = depth_fs_feat[::-1]
                distill_feature = []
                for i, adapt in enumerate(self.adapt_list):
                    distill_feature.append(self.__getattr__(adapt)(rgb_fs_feat[i]))
                
                ### distillation loss
                head_loss, _ = compute_head_distill_loss(rgb_outputs, depth_outputs, target)
                backbone_loss_affinity = compute_backbone_resize_affinity_loss(distill_feature, depth_fs_feat)
                relation_loss = spearman_loss(distill_feature, depth_fs_feat)

                return rgb_outputs, relation_loss, backbone_loss_affinity, head_loss

        else:
            rgb = input['rgb']
            rgb_feat, rgb_outputs, fusion_features = self.centernet_rgb(rgb, coord_ranges, calibs, target, mode='val')

            return rgb_feat, rgb_outputs, rgb_outputs

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



