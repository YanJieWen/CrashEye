'''
@File: build.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 28, 2025
@HomePage: https://github.com/YanJieWen
'''

import importlib
import os
import os.path as osp
from glob import glob
#openmmlab-based
# from modeling.mm_detector import MMdetector
# from modeling.mm_tracker import MMTracker


model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = glob(osp.join(model_folder,'*.py'))
model_modules = [importlib.import_module(f'modeling.{osp.basename(x).split(".")[0]}') for x in model_filenames if 'detector' in x or 'tracker' in x]
#all cfg
def build_detector(cfg):
    detector = None
    if cfg.Track.model == 'openmm' and 'mix' not in cfg.task_name:
        for m in model_modules:
            detector = getattr(m,'MMdetector',None)
            if detector is not None:
                break
            else:
                pass
        detector = detector(cfg.Track.detector, cfg.task_name, cfg.detect.train.work_dir)
    elif cfg.Track.model =='masort' and 'mix' not in cfg.task_name:
        for m in model_modules:
            detector  = getattr(m,'ULTdetector',None)
            if detector is not None:
                break
            else:
                pass
        detector = detector(cfg.Track.detector)
    elif cfg.task_name.split('_')[1]=='mm':
        for m in model_modules:
            detector = getattr(m,'MMdetector',None)
            if detector is not None:
                break
            else:
                pass
        detector = detector(cfg.Track.detector, cfg.task_name, cfg.detect.train.work_dir)
    elif cfg.task_name.split('_')[1] == 'ult':
        for m in model_modules:
            detector = getattr(m, 'ULTdetector', None)
            if detector is not None:
                break
            else:
                pass
        detector = detector(cfg.Track.detector)
    if detector is None:
        raise ValueError(f'module is not defined!')
    return detector
#track_cfg
def build_tracker(cfg):
    tracker = None
    if cfg.model == 'openmm':
        for m in model_modules:
            tracker = getattr(m,'MMTracker',None)
            if tracker is not None:
                break
            else:
                pass
    elif cfg.model =='masort':
        for m in model_modules:
            tracker = getattr(m,'ULTtracker',None)
            if tracker is not None:
                break
            else:
                pass
    else: #todo: mix framework
        pass
    if tracker is None:
        raise ValueError(f'module is not defined!')
    tracker = tracker(cfg.tracker)
    return tracker


# if __name__ == '__main__':
#     from utils import yaml_load, Config
#
#     config = yaml_load('./configs/mm_mot.yaml')
#     cfg = Config(config)
#     detector = build_detector(cfg)
#     tracker = build_tracker(cfg)
#     print(detector)
#     print(tracker)


