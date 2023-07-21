import torch
import random
import numpy as np

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester


parser = argparse.ArgumentParser(description='implementation of MonoSKD')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--config', type=str, default='experiments/config.yaml')
args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed**2)
    torch.manual_seed(seed**3)
    torch.cuda.manual_seed(seed**4)
    torch.cuda.manual_seed_all(seed**4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'], 'train.log'))

    #  build dataloader
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])

    # evaluation mode
    if args.evaluate:
        logger.info('###################  Testing  ##################')
        model = build_model(cfg['model'], train_loader.dataset.cls_mean_size, 'testing')
        tester = Tester(cfg['tester'], cfg['dataset'], model, val_loader, logger)
        tester.test()
        return

    # build model
    model = build_model(cfg['model'], train_loader.dataset.cls_mean_size, 'training')
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr & bnm scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    logger.info('###################  Training  ##################')
    logger.info('KD type: %s' % (str(cfg['model']['kd_type'])))
    logger.info('Batch Size: %d'  % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f'  % (cfg['optimizer']['lr']))
    logger.info('Max Epoch: %d'  % (cfg['trainer']['max_epoch']))
    logger.info('Model Type: %s'  % (cfg['model']['type']))
    logger.info('Writelist: ' + str(cfg['dataset']['writelist']))
    logger.info('Log_dir: %s' % (cfg['trainer']['log_dir']))
    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger)
    trainer.train()


if __name__ == '__main__':
    main()