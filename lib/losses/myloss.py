import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import einsum
from lib.losses.feature_distill_loss import calculate_box_mask
import torchvision.ops.roi_align as roi_align
import torchsort


def corrcoef(target, pred):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

def spearman_loss(
    target,
    pred,
    regularization="l2",
    regularization_strength=1.0,
):
    spearman_loss = 0.0
    resize_shape = pred[-1].shape[-2:]  # save training time

    if isinstance(pred, list):
        for i in range(len(pred)):
            feature_target = target[i]
            feature_pred = pred[i]
            B, C, H, W = feature_pred.shape

            feature_pred_down = F.interpolate(feature_pred,
                                              size=resize_shape,
                                              mode="bilinear")
            feature_target_down = F.interpolate(feature_target,
                                                size=resize_shape,
                                                mode="bilinear")

            feature_pred_down = feature_pred_down.reshape(B, -1)
            feature_target_down = feature_target_down.reshape(B, -1)

            feature_pred_down = torchsort.soft_rank(
                feature_pred_down,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )
            spearman_loss += 1 - corrcoef(
                feature_target_down,
                feature_pred_down / feature_pred_down.shape[-1])
    return spearman_loss