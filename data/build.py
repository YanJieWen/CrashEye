# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

# from torch.utils import data
#
# from .datasets.mnist import MNIST
# from .transforms import build_transforms

# DATASETS = {
#     'mot': 'MOTDataset',
# }

# def build_dataset(transforms, is_train=True):
#     datasets = MNIST(root='./', train=is_train, transform=transforms, download=True)
#     return datasets
#
#
# def make_data_loader(cfg, is_train=True):
#     if is_train:
#         batch_size = cfg.SOLVER.IMS_PER_BATCH
#         shuffle = True
#     else:
#         batch_size = cfg.TEST.IMS_PER_BATCH
#         shuffle = False
#
#     transforms = build_transforms(cfg, is_train)
#     datasets = build_dataset(transforms, is_train)
#
#     num_workers = cfg.DATALOADER.NUM_WORKERS
#     data_loader = data.DataLoader(
#         datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
#     )
#
#     return data_loader



import importlib
import os
import os.path as osp
from glob import glob

import numpy as np
import torch.utils.data
from utils import yaml_load,Config
from data.transforms import ValTransform

# config = yaml_load('./configs/mm_mot.yaml')
# cfg = Config(config)
# track_cfg = cfg.Track
# track_data_cfg = track_cfg.dataset



# def build_dataloder
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = glob(os.path.join(model_folder,'datasets/*.py'))
_model_modules = [importlib.import_module(f'data.datasets.{osp.basename(x).split(".")[0]}') for x in model_filenames]

def build_dataloader(cfg):
    model_name = cfg.data_type
    dataset = None
    for m in _model_modules:
        dataset = getattr(m,model_name,None)
        if dataset is not None:
            break
        else:
            pass
    if dataset is None:
        raise ValueError(f'{model_name} is not found')
    cfg_dict = cfg.__dict__
    cfg_dict.pop('data_type')
    cfg_dict['img_size'] = eval(cfg_dict['img_size'])
    # transform = ValTransform(rgb_means=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),)
    transform = ValTransform()
    cfg_dict['preproc'] = transform
    dataset = dataset(**cfg_dict)
    #build dataloader
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader_kwargs = {
                "num_workers": 0,
                "pin_memory": True,
                "sampler": sampler,
            }
    dataloader_kwargs["batch_size"] = 1
    val_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return val_loader

#
# if __name__ == '__main__':
#     import torchvision.transforms as ts
#     import cv2
#     config = yaml_load('./configs/mm_mot.yaml')
#     cfg = Config(config)
#     track_cfg = cfg.Track
#     cfg = track_cfg.dataset
#     data_loader=build_dataloader(cfg)
#     # print(data_loader[0][0].shape)
#     img = np.asarray(next(iter(data_loader))[0][0].numpy().transpose(1,2,0),dtype=np.uint8)[...,::-1]
#     print(img)
#     cv2.imshow('demo',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


