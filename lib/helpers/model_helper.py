from lib.models.DID import DID
from lib.models.DID_distill import DID_Distill


def build_model(cfg, mean_size, flag):
    if cfg['type'] == 'DID':
        return DID(backbone=cfg['backbone'],
                   neck=cfg['neck'],
                   mean_size=mean_size,
                   model_type='DID')
    elif cfg['type'] == 'distill':
        return DID_Distill(backbone=cfg['backbone'],
                           neck=cfg['neck'],
                           flag=flag,
                           mean_size=mean_size,
                           model_type='distill',
                           cfg=cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
