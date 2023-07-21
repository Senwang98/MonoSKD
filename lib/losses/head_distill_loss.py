import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.helpers.decode_helper import _transpose_and_gather_feat

from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian


def calculate_box_mask_gaussian(preds_shape, target, downsample_ratio):
    B, C, H, W = preds_shape
    gt_mask = np.zeros((B, H, W), dtype=np.float32)  # C * H * W

    for i in range(B):
        for j in range(target['obj_num'][i]):
            bbox2d = target['box2d_gt_head'][i, j] / downsample_ratio
            bbox2d_gt = bbox2d.cpu()
            left_top = bbox2d_gt[:2]
            right_bottom = bbox2d_gt[2:]

            w, h = right_bottom[1] - left_top[1], right_bottom[0] - left_top[0]
            radius = gaussian_radius((w, h))
            radius = max(0, int(radius))

            center_heatmap = [int((right_bottom[0] + left_top[0]) / 2), int((right_bottom[1] + left_top[1]) / 2)]
            draw_umich_gaussian(gt_mask[i], center_heatmap, radius)

    gt_mask_torch = torch.from_numpy(gt_mask)
    gt_mask_torch = (gt_mask_torch > 0).float()
    return gt_mask_torch

def calculate_depth_mask(preds_shape, target, downsample_ratio):
    B, C, H, W = preds_shape
    gt_mask = torch.zeros((B, H, W))

    for i in range(B):
        for j in range(target['obj_num'][i]):
            bbox2d_gt = target['box2d_gt'][i, j] / downsample_ratio

            left_top = bbox2d_gt[:2]
            right_bottom = bbox2d_gt[2:]

            left_top_x = int(left_top[0].item())
            left_top_y = int(left_top[1].item())

            right_bottom_x = int(right_bottom[0].item())
            right_bottom_y = int(right_bottom[1].item())

            gt_mask[i, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 1
    return gt_mask

def compute_head_distill_loss(rgb_output, depth_output, target):
    stats_dict = {}
    shape = depth_output['offset_2d'].shape
    mask = calculate_box_mask_gaussian(shape, target, 4)

    if rgb_output['train_tag'].sum() > 0:
        # heatmap_distill_loss = compute_head_loss(rgb_output['heatmap'], depth_output['heatmap'], mask)
        offset2d_distill_loss = compute_head_loss(rgb_output['offset_2d'], depth_output['offset_2d'], mask)
        size2d_distill_loss = compute_head_loss(rgb_output['size_2d'], depth_output['size_2d'], mask)
        offset3d_distill_loss = compute_DID_head_loss(rgb_output['offset_3d'], depth_output['offset_3d'], mask)
        size3d_distill_loss = compute_DID_head_loss(rgb_output['size_3d'], depth_output['size_3d'], mask)
        heading_distill_loss = compute_DID_heading_distill_loss(rgb_output, depth_output, target)

        # modified
        vis_depth_pred = rgb_output['vis_depth']
        vis_depth_gt = depth_output['vis_depth'].detach()
        vis_depth_distill_loss = compute_DID_head_loss(vis_depth_pred, vis_depth_gt, mask)

        att_depth_pred = rgb_output['att_depth']
        att_depth_gt = depth_output['att_depth'].detach()
        att_depth_distill_loss = compute_DID_head_loss(att_depth_pred, att_depth_gt, mask)

        ins_depth_pred = rgb_output['ins_depth']
        ins_depth_gt = depth_output['ins_depth'].detach()
        ins_depth_distill_loss = compute_DID_head_loss(ins_depth_pred, ins_depth_gt, mask)

        depth_distill_loss = vis_depth_distill_loss + att_depth_distill_loss + ins_depth_distill_loss

        stats_dict['offset2d'] = offset2d_distill_loss.item()
        stats_dict['size2d'] = size2d_distill_loss.item()
        stats_dict['offset3d'] = offset3d_distill_loss.item()
        stats_dict['depth'] = depth_distill_loss.item()
        stats_dict['size3d'] = size3d_distill_loss.item()
        stats_dict['heading'] = heading_distill_loss.item()
        # stats_dict['heatmap'] = heatmap_distill_loss.item()

        total_distill_loss = offset2d_distill_loss + size2d_distill_loss + offset3d_distill_loss + \
                    depth_distill_loss + size3d_distill_loss + heading_distill_loss 
                    # + heatmap_distill_loss
        return total_distill_loss, stats_dict
    else:
        stats_dict['offset2d'] = 0.
        stats_dict['size2d'] = 0.
        stats_dict['offset3d'] = 0.
        stats_dict['depth'] = 0.
        stats_dict['size3d'] = 0.
        stats_dict['heading'] = 0.
        # stats_dict['heatmap'] = 0.
        total_distill_loss = torch.tensor(0.0).to(depth_output['offset_2d'].device)
        return total_distill_loss, stats_dict


def compute_head_loss(pred, gt, mask):
    pred = pred.permute(0, *range(2, len(pred.shape)), 1) # [8, 96, 320, 2]
    gt = gt.permute(0, *range(2, len(gt.shape)), 1) # [8, 96, 320, 2]

    positives = pred.new_ones(*pred.shape[:3]) # [8, 96, 320]
    positives = positives * mask.cuda()

    reg_weights = positives.float()

    pos_inds = reg_weights > 0
    pos_feature_preds = pred[pos_inds]
    pos_feature_targets = gt[pos_inds]

    head_distill_loss = F.l1_loss(pos_feature_preds, pos_feature_targets, reduction='mean')
    return head_distill_loss

def compute_DID_head_loss(pred, gt, mask=None):
    head_distill_loss = F.l1_loss(pred, gt, reduction='mean')
    return head_distill_loss

def compute_heading_distill_loss(rgb_output, depth_output, target):
    heading_rgb_input = _transpose_and_gather_feat(rgb_output['heading'], target['indices'])  # B * C * H * W ---> B * K * C
    heading_rgb_input = heading_rgb_input.view(-1, 24)

    heading_depth_input = _transpose_and_gather_feat(depth_output['heading'], target['indices'])  # B * C * H * W ---> B * K * C
    heading_depth_input = heading_depth_input.view(-1, 24)
    heading_target_cls = target['heading_bin'].view(-1)
    # classification loss
    heading_input_rgb_cls = heading_rgb_input[:, 0:12]
    heading_input_depth_cls = heading_depth_input[:, 0:12]

    cls_distill_loss = F.kl_div(heading_input_rgb_cls.softmax(dim=-1).log(), heading_input_depth_cls.softmax(dim=-1), reduction='mean')

    # regression loss
    heading_rgb_input_res = heading_rgb_input[:, 12:24]
    heading_depth_input_res = heading_depth_input[:, 12:24]

    cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1,
                                                                              index=heading_target_cls.view(-1, 1),
                                                                              value=1)

    heading_rgb_input_res = torch.sum(heading_rgb_input_res * cls_onehot, 1)
    heading_depth_input_res = torch.sum(heading_depth_input_res * cls_onehot, 1)
    reg_distill_loss = F.l1_loss(heading_rgb_input_res, heading_depth_input_res, reduction='mean')
    return cls_distill_loss + reg_distill_loss

def compute_DID_heading_distill_loss(rgb_output, depth_output, target):
    mask = target['mask_2d'].view(-1)
    heading_rgb_input   = rgb_output['heading'][rgb_output['train_tag']]    # num_bin * 24
    heading_rgb_input   = heading_rgb_input.view(-1, 24)
    heading_depth_input = depth_output['heading'][rgb_output['train_tag']]  # num_bin * 24
    heading_depth_input = heading_depth_input.view(-1, 24)

    # cls
    heading_target_cls = target['heading_bin'].view(-1)
    heading_target_cls  = heading_target_cls[mask]

    # classification loss
    heading_input_rgb_cls = heading_rgb_input[:, 0:12]
    heading_input_depth_cls = heading_depth_input[:, 0:12]
    cls_distill_loss = F.kl_div(heading_input_rgb_cls.softmax(dim=-1).log(), heading_input_depth_cls.softmax(dim=-1), reduction='mean')

    # regression loss
    heading_rgb_input_res = heading_rgb_input[:, 12:24]
    heading_depth_input_res = heading_depth_input[:, 12:24]

    cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)

    heading_rgb_input_res = torch.sum(heading_rgb_input_res * cls_onehot, 1)
    heading_depth_input_res = torch.sum(heading_depth_input_res * cls_onehot, 1)
    reg_distill_loss = F.l1_loss(heading_rgb_input_res, heading_depth_input_res, reduction='mean') # if input is [], then loss = NaN

    return cls_distill_loss + reg_distill_loss
