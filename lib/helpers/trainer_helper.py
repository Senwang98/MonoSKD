import os
import tqdm

import torch
import torch.nn as nn
import numpy as np
import pdb
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.save_helper import load_checkpoint
from lib.losses.loss_function import DIDLoss,Hierarchical_Task_Learning
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

from tools import eval


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        
        self.cfg = cfg
        self.model_type=cfg['model']['type']
        self.kd_type=cfg['model']['kd_type']

        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        self.label_dir = cfg['dataset']['label_dir']
        self.eval_cls = cfg['dataset']['eval_cls']

        # loading pretrain/resume model
        if self.model_type == 'DID':
            if self.cfg_train.get('pretrain_model'):
                assert os.path.exists(self.cfg_train['pretrain_model'])
                load_checkpoint(model=self.model, optimizer=None, filename=self.cfg_train['pretrain_model'], map_location=self.device, logger=self.logger)

            if self.cfg_train.get('resume_model', None):
                assert os.path.exists(self.cfg_train['resume_model'])
                self.epoch = load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger, map_location=self.device)
                self.lr_scheduler.last_epoch = self.epoch - 1
        
        elif self.model_type == 'distill':
            if self.cfg_train.get('pretrain_model'):
                if os.path.exists(self.cfg_train['pretrain_model']['rgb']):
                    load_checkpoint(model=self.model.centernet_rgb, optimizer=None, filename=self.cfg_train['pretrain_model']['rgb'], map_location=self.device, logger=self.logger)
                else:
                    self.logger.info("no rgb pretrained model")
                    assert os.path.exists(self.cfg_train['pretrain_model']['rgb'])

                if os.path.exists(self.cfg_train['pretrain_model']['depth']):
                    load_checkpoint(model=self.model.centernet_depth, optimizer=None, filename=self.cfg_train['pretrain_model']['depth'], map_location=self.device, logger=self.logger)
                else:
                    self.logger.info("no depth pretrained model")
                    assert os.path.exists(self.cfg_train['pretrain_model']['depth'])

            if self.cfg_train.get('resume_model', None):
                assert os.path.exists(self.cfg_train['resume_model'])
                self.epoch = load_checkpoint(model=self.model, optimizer=self.optimizer, filename=self.cfg_train['resume_model'], map_location=self.device, logger=self.logger)
                self.lr_scheduler.last_epoch = self.epoch - 1

        self.model = torch.nn.DataParallel(model).to(self.device)

    def train(self):
        start_epoch = self.epoch
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss)
        for epoch in range(start_epoch, self.cfg_train['max_epoch']):
            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' %(epoch + 1))
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
            else:
                self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss,self.epoch)

            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' %(key[:-4], loss_weights[key])
            self.logger.info(log_str)

            # ei_loss = self.train_one_epoch(loss_weights)
            if self.model_type == 'DID':
                ei_loss = self.train_one_epoch(loss_weights)

            elif self.model_type == 'distill':
                ei_loss = self.train_one_epoch_distill(loss_weights)

            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if ((self.epoch % self.cfg_train['eval_frequency']) == 0 and \
                self.epoch >= self.cfg_train['eval_start']):
                self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                self.eval_one_epoch()


            if ((self.epoch % self.cfg_train['save_frequency']) == 0
                and self.epoch >= self.cfg_train['eval_start']):
                os.makedirs(self.cfg_train['log_dir']+'/checkpoints', exist_ok=True)
                ckpt_name = os.path.join(self.cfg_train['log_dir']+'/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)

        return None

    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():
            for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)

                # train one batch
                # criterion = DIDLoss(self.epoch)
                # outputs = self.model(inputs,coord_ranges,calibs,targets)
                # _, loss_terms = criterion(outputs, targets)

                # train one batch
                if self.model_type == 'DID':
                    criterion = DIDLoss(self.epoch)
                    _, outputs, _ = self.model(inputs,coord_ranges,calibs,targets)
                    _, loss_terms = criterion(outputs, targets)
                elif self.model_type == 'distill':
                    # rgb_outputs, backbone_loss_l1, backbone_loss_affinity, head_loss, _ = self.model(inputs, coord_ranges, calibs, targets, self.epoch)
                    rgb_outputs, relation_loss, backbone_loss_affinity, head_loss = self.model(inputs, coord_ranges, calibs, targets, self.epoch)
                    criterion = DIDLoss(self.epoch)
                    _, loss_terms = criterion(rgb_outputs, targets)

                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch
        return disp_dict

    def train_one_epoch(self,loss_weights=None):
        self.model.train()

        disp_dict = {}
        stat_dict = {}
        for batch_idx, (inputs, calibs, coord_ranges, targets, info) in enumerate(self.train_loader):
            if type(inputs) != dict:
                inputs = inputs.to(self.device)
            else:
                for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            for key in targets.keys(): targets[key] = targets[key].to(self.device)
            # train one batch
            self.optimizer.zero_grad()
            criterion = DIDLoss(self.epoch)
            feat_backbone, outputs, fusion_features = self.model(inputs,coord_ranges,calibs,targets)

            total_loss, loss_terms = criterion(outputs, targets)

            if loss_weights is not None:
                total_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    total_loss += loss_weights[key].detach()*loss_terms[key]
            total_loss.backward()
            self.optimizer.step()

            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0

                if isinstance(loss_terms[key], int):
                    stat_dict[key] += (loss_terms[key])
                else:
                    stat_dict[key] += (loss_terms[key]).detach()
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                # disp_dict[key] += loss_terms[key]
                if isinstance(loss_terms[key], int):
                    disp_dict[key] += (loss_terms[key])
                else:
                    disp_dict[key] += (loss_terms[key]).detach()
            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' %(key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)

        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch

        return stat_dict

    def train_one_epoch_distill(self, loss_weights=None):
        self.model.train()
        disp_dict = {}
        stat_dict = {}

        loss_stats = ['rgb_loss']
        if 'fg_kd' in self.kd_type :
            loss_stats.append('backbone_loss_l1')
        if 'affinity_kd' in self.kd_type:
            loss_stats.append('backbone_loss_affinity')
        if 'head_kd' in self.kd_type:
            loss_stats.append('head_loss')
        if 'spearman_kd' in self.kd_type:
            loss_stats.append('Spearman_loss')

        for batch_idx, (inputs, calibs, coord_ranges, targets, info) in enumerate(self.train_loader):
            # inputs = inputs.to(self.device)
            for key in inputs.keys(): 
                inputs[key] = inputs[key].to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            for key in targets.keys(): 
                targets[key] = targets[key].to(self.device)
            
            # train one batch
            self.optimizer.zero_grad()
            
            if 'spearman_kd' in self.kd_type:
                rgb_outputs, relation_loss, backbone_loss_affinity, head_loss = self.model(inputs, coord_ranges, calibs, targets)
            else:
                rgb_outputs, backbone_loss_l1, backbone_loss_affinity, head_loss = self.model(inputs, coord_ranges, calibs, targets)
            
            criterion = DIDLoss(self.epoch)
            rgb_loss, loss_terms = criterion(rgb_outputs, targets)
            if loss_weights is not None:
                rgb_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    rgb_loss += loss_weights[key].detach()*loss_terms[key]
            
            rgb_loss = rgb_loss.mean()
            total_loss = rgb_loss

            if 'fg_kd' in self.kd_type:
                backbone_loss_l1 = (10 * backbone_loss_l1).mean()
                if 'backbone_loss_l1' not in stat_dict.keys(): 
                    stat_dict['backbone_loss_l1'] = 0
                stat_dict['backbone_loss_l1'] += backbone_loss_l1

                if 'backbone_loss_l1' not in disp_dict.keys():
                    disp_dict['backbone_loss_l1'] = 0
                disp_dict['backbone_loss_l1'] += backbone_loss_l1
            
                total_loss = total_loss + backbone_loss_l1

            if 'affinity_kd' in self.kd_type:
                backbone_loss_affinity = backbone_loss_affinity.mean()
                if 'affinity_kd' not in stat_dict.keys(): 
                    stat_dict['affinity_kd'] = 0
                stat_dict['affinity_kd'] += backbone_loss_affinity

                if 'affinity_kd' not in disp_dict.keys():
                    disp_dict['affinity_kd'] = 0
                disp_dict['affinity_kd'] += backbone_loss_affinity
                total_loss = total_loss + backbone_loss_affinity

            if 'head_kd' in self.kd_type:
                head_loss = head_loss.mean()
                if 'head_kd' not in stat_dict.keys(): 
                    stat_dict['head_kd'] = 0
                stat_dict['head_kd'] += head_loss

                if 'head_kd' not in disp_dict.keys():
                    disp_dict['head_kd'] = 0
                disp_dict['head_kd'] += head_loss
                total_loss = total_loss + head_loss
            
            if 'spearman_kd' in self.kd_type:
                relation_loss = relation_loss.mean()
                if 'spearman_kd' not in stat_dict.keys(): 
                    stat_dict['spearman_kd'] = 0
                stat_dict['spearman_kd'] += relation_loss

                if 'spearman_kd' not in disp_dict.keys():
                    disp_dict['spearman_kd'] = 0
                disp_dict['spearman_kd'] += relation_loss
                total_loss = total_loss + relation_loss

            total_loss.backward()
            self.optimizer.step()
            
            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0
                stat_dict[key] += loss_terms[key] 
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += loss_terms[key]

            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' %(key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)
                
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
                            
        return stat_dict

    def eval_one_epoch(self):
        self.model.eval()

        results = {}
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)

                _, outputs, _ = self.model(inputs,coord_ranges,calibs,K=50,mode='val')

                dets = extract_dets_from_outputs(outputs, K=50)
                dets = dets.detach().cpu().numpy()

                # get corresponding calibs & transform tensor to numpy
                calibs = [self.test_loader.dataset.get_calib(index)  for index in info['img_id']]
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.test_loader.dataset.cls_mean_size
                dets = decode_detections(dets = dets,
                                        info = info,
                                        calibs = calibs,
                                        cls_mean_size=cls_mean_size,
                                        threshold = self.cfg_test['threshold'])
                results.update(dets)
                progress_bar.update()
            progress_bar.close()
        # self.save_results(results)
        out_dir = os.path.join(self.cfg_train['out_dir'], 'EPOCH_' + str(self.epoch))
        self.save_results(results, out_dir)
        eval.eval_from_scrach(
            self.label_dir,
            os.path.join(out_dir, 'data'),
            self.eval_cls,
            ap_mode=40,
            logger=self.logger)

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()
