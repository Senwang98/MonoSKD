import torch
from collections import OrderedDict

# You can use this script to get student ckpt

def change_model(output_path=None):
    stu_model = torch.load('kitti_models/logs/monoskd/checkpoints/checkpoint_epoch_140.pth')
    all_name = []
    for name, v in stu_model["model_state"].items():
        if 'centernet_rgb' in name:
            tmp_name = name.replace('centernet_rgb.', '')
            all_name.append((tmp_name, v))
        else:
            continue
    state_dict = OrderedDict(all_name)
    print(state_dict.keys())
    stu_model['model_state'] = state_dict
    stu_model.pop('optimizer_state')
    stu_model.pop('epoch')
    torch.save(stu_model, output_path)

change_model('distill_student_mAP20.21.pth')
