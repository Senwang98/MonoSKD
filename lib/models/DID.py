import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from lib.backbones.resnet import resnet50
from lib.backbones.dla import dla34
from lib.backbones.dlaup import DLAUp
from lib.backbones.dlaup import DLAUpv2

import torchvision.ops.roi_align as roi_align
from lib.losses.loss_function import extract_input_from_tensor
from lib.helpers.decode_helper import _topk,_nms



def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class DID(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', downsample=4, mean_size=None, model_type='DID'):
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.model_type = model_type
        self.backbone = globals()[backbone](pretrained=True, return_levels=True)
        self.head_conv = 256  # default setting for head conv
        self.mean_size = nn.Parameter(torch.tensor(mean_size,dtype=torch.float32),requires_grad=False)
        self.cls_num = mean_size.shape[0]
        channels = self.backbone.channels  # channels list for feature maps generated by backbone

        self.first_level = int(np.log2(downsample))
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.feat_up = globals()[neck](channels[self.first_level:], scales_list=scales)

        # initialize the head of pipeline, according to heads setting.
        self.heatmap = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 3, kernel_size=1, stride=1, padding=0, bias=True))
        self.offset_2d = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.size_2d = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))

        self.offset_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.size_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 3, kernel_size=1, stride=1, padding=0, bias=True))
        self.heading = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 24, kernel_size=1, stride=1, padding=0, bias=True))

        self.vis_depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        self.att_depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        self.vis_depth_uncer = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        self.att_depth_uncer = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))


        # init layers
        self.heatmap[-1].bias.data.fill_(-2.19)
        self.fill_fc_weights(self.offset_2d)
        self.fill_fc_weights(self.size_2d)

        self.offset_3d.apply(weights_init_xavier)
        self.size_3d.apply(weights_init_xavier)
        self.heading.apply(weights_init_xavier)

        self.vis_depth.apply(weights_init_xavier)
        self.att_depth.apply(weights_init_xavier)
        self.vis_depth_uncer.apply(weights_init_xavier)
        self.att_depth_uncer.apply(weights_init_xavier)


    def forward(self, input, coord_ranges,calibs, targets=None, K=50, mode='train'):

        if self.model_type == 'DID':
            input = input['rgb']
            # input = input['depth']

        device_id = input.device
        feat_backbone = self.backbone(input)
        fusion_features = self.feat_up(feat_backbone[self.first_level:])
        feat = fusion_features[-1]
        ret = {}
        '''
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        '''
        ret['heatmap']=self.heatmap(feat)
        ret['offset_2d']=self.offset_2d(feat)
        ret['size_2d']=self.size_2d(feat)

        # torch.cuda.synchronize()
        #two stage
        # assert(mode in ['train','val','test'])
        assert(mode in ['train','val','test'])
        if mode=='train':   #extract train structure in the train (only) and the val mode
            inds,cls_ids = targets['indices'],targets['cls_ids']
            masks = targets['mask_2d']
            # masks = targets['mask_2d'].type(torch.bool)
        else:    #extract test structure in the test (only) and the val mode
            inds,cls_ids = _topk(_nms(torch.clamp(ret['heatmap'].sigmoid(), min=1e-4, max=1 - 1e-4)), K=K)[1:3]
            # if torch.__version__ == '1.10.0+cu113':
            if torch.__version__ in ['1.10.0+cu113', '1.10.0', '1.6.0', '1.4.0']:
                masks = torch.ones(inds.size()).type(torch.bool).to(device_id)
            else:
                masks = torch.ones(inds.size()).type(torch.uint8).to(device_id)

        ret.update(self.get_roi_feat(feat,inds,masks,ret,calibs,coord_ranges,cls_ids))
        return feat_backbone, ret, fusion_features


    def get_roi_feat_by_mask(self,feat,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids):
        BATCH_SIZE,_,HEIGHT,WIDE = feat.size()
        device_id = feat.device
        num_masked_bin = mask.sum()
        res = {}

        if num_masked_bin!=0:
            #get box2d of each roi region
            scale_box2d_masked = extract_input_from_tensor(box2d_maps,inds,mask)
            #get roi feature
            roi_feature_masked = roi_align(feat,scale_box2d_masked,[7,7])
            #get coord range of each roi
            coord_ranges_mask2d = coord_ranges[scale_box2d_masked[:,0].long()]

            #map box2d coordinate from feature map size domain to original image size domain
            box2d_masked = torch.cat([scale_box2d_masked[:,0:1],
                       scale_box2d_masked[:,1:2]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
                       scale_box2d_masked[:,2:3]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2],
                       scale_box2d_masked[:,3:4]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
                       scale_box2d_masked[:,4:5]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2]],1)
            roi_calibs = calibs[box2d_masked[:,0].long()]
            #project the coordinate in the normal image to the camera coord by calibs
            coords_in_camera_coord = torch.cat([self.project2rect(roi_calibs,torch.cat([box2d_masked[:,1:3],torch.ones([num_masked_bin,1]).to(device_id)],-1))[:,:2],
                                          self.project2rect(roi_calibs,torch.cat([box2d_masked[:,3:5],torch.ones([num_masked_bin,1]).to(device_id)],-1))[:,:2]],-1)
            coords_in_camera_coord = torch.cat([box2d_masked[:,0:1],coords_in_camera_coord],-1)
            #generate coord maps
            coord_maps = torch.cat([torch.cat([coords_in_camera_coord[:,1:2]+i*(coords_in_camera_coord[:,3:4]-coords_in_camera_coord[:,1:2])/(7-1) for i in range(7)],-1).unsqueeze(1).repeat([1,7,1]).unsqueeze(1),
                                torch.cat([coords_in_camera_coord[:,2:3]+i*(coords_in_camera_coord[:,4:5]-coords_in_camera_coord[:,2:3])/(7-1) for i in range(7)],-1).unsqueeze(2).repeat([1,1,7]).unsqueeze(1)],1)

            #concatenate coord maps with feature maps in the channel dim
            cls_hots = torch.zeros(num_masked_bin,self.cls_num).to(device_id)
            cls_hots[torch.arange(num_masked_bin).to(device_id),cls_ids[mask].long()] = 1.0
            
            roi_feature_masked = torch.cat([roi_feature_masked,coord_maps,cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1,1,7,7])],1)

            #compute 3d dimension offset
            size3d_offset = self.size_3d(roi_feature_masked)[:,:,0,0]

            #compute scale factor
            scale_depth = torch.clamp((scale_box2d_masked[:,4]-scale_box2d_masked[:,2])*4, min=1.0) / \
                          torch.clamp(box2d_masked[:,4]-box2d_masked[:,2], min=1.0)

            vis_depth = self.vis_depth(roi_feature_masked).squeeze(1)
            att_depth = self.att_depth(roi_feature_masked).squeeze(1)

            vis_depth = (-vis_depth).exp()
            vis_depth = vis_depth * scale_depth.unsqueeze(-1).unsqueeze(-1)

            vis_depth_uncer = self.vis_depth_uncer(roi_feature_masked)[:, 0, :, :]
            att_depth_uncer = self.att_depth_uncer(roi_feature_masked)[:, 0, :, :]

            ins_depth = vis_depth + att_depth
            ins_depth_uncer = torch.logsumexp(torch.stack([vis_depth_uncer, att_depth_uncer], -1), -1)

            res['train_tag'] = torch.ones(num_masked_bin).type(torch.bool).to(device_id)
            res['heading'] = self.heading(roi_feature_masked)[:,:,0,0]

            res['vis_depth'] = vis_depth
            res['att_depth'] = att_depth
            res['ins_depth'] = ins_depth

            res['vis_depth_uncer'] = vis_depth_uncer
            res['att_depth_uncer'] = att_depth_uncer
            res['ins_depth_uncer'] = ins_depth_uncer


            res['offset_3d'] = self.offset_3d(roi_feature_masked)[:,:,0,0]
            res['size_3d']= size3d_offset

        else:
            res['offset_3d'] = torch.zeros([1,2]).to(device_id)
            res['size_3d'] = torch.zeros([1,3]).to(device_id)
            res['train_tag'] = torch.zeros(1).type(torch.bool).to(device_id)
            res['heading'] = torch.zeros([1,24]).to(device_id)


            res['vis_depth'] = torch.zeros([1,7,7]).to(device_id)
            res['att_depth'] = torch.zeros([1,7,7]).to(device_id)
            res['ins_depth'] = torch.zeros([1,7,7]).to(device_id)

            res['vis_depth_uncer'] = torch.zeros([1,7,7]).to(device_id)
            res['att_depth_uncer'] = torch.zeros([1,7,7]).to(device_id)
            res['ins_depth_uncer'] = torch.zeros([1,7,7]).to(device_id)


        return res


    def get_roi_feat(self,feat,inds,mask,ret,calibs,coord_ranges,cls_ids):
        BATCH_SIZE,_,HEIGHT,WIDE = feat.size()
        device_id = feat.device
        coord_map = torch.cat([torch.arange(WIDE).unsqueeze(0).repeat([HEIGHT,1]).unsqueeze(0),\
                        torch.arange(HEIGHT).unsqueeze(-1).repeat([1,WIDE]).unsqueeze(0)],0).unsqueeze(0).repeat([BATCH_SIZE,1,1,1]).type(torch.float).to(device_id)
        box2d_centre = coord_map + ret['offset_2d']
        box2d_maps = torch.cat([box2d_centre-ret['size_2d']/2,box2d_centre+ret['size_2d']/2],1)
        box2d_maps = torch.cat([torch.arange(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat([1,1,HEIGHT,WIDE]).type(torch.float).to(device_id),box2d_maps],1)
        #box2d_maps is box2d in each bin
        res = self.get_roi_feat_by_mask(feat,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids)
        return res

    def project2rect(self,calib,point_img):
        c_u = calib[:,0,2]
        c_v = calib[:,1,2]
        f_u = calib[:,0,0]
        f_v = calib[:,1,1]
        b_x = calib[:,0,3]/(-f_u) # relative
        b_y = calib[:,1,3]/(-f_v)
        x = (point_img[:,0]-c_u)*point_img[:,2]/f_u + b_x
        y = (point_img[:,1]-c_v)*point_img[:,2]/f_v + b_y
        z = point_img[:,2]
        centre_by_obj = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)],-1)
        return centre_by_obj

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    import torch
    net = CenterNet3D()
    print(net)
    input = torch.randn(4, 3, 384, 1280)
    print(input.shape, input.dtype)
    output = net(input)