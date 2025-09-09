'''
@File: ult_detector.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 9月 01, 2025
@HomePage: https://github.com/YanJieWen
'''

import torch
import copy
import numpy as np
import os
import os.path as osp
from glob import glob
from ultralytics import YOLO

class ULTdetector():
    def __init__(self,cfg):
        if hasattr(cfg,'imgsz'):
            cfg.imgsz = eval(cfg.imgsz)
        self.model = YOLO(cfg.ckpt)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.det_cfg = cfg
    def __call__(self, img):
        if len(img.shape)==4:
            img = img[0]
            if isinstance(img,torch.Tensor):
                img = np.asarray(img.cpu().numpy().transpose(1, 2, 0) * 255., dtype=np.uint8)[..., ::-1]
        else:
            pass
        det_dict = copy.deepcopy(self.det_cfg.__dict__)
        det_dict.pop("ckpt")
        det_dict['source'] = img

        results = self.model(**det_dict,verbose=False)
        detected_bboxes = results[0].boxes
        out_bboxes = []
        clss =detected_bboxes.cls.cpu().numpy()
        confs = detected_bboxes.conf.cpu().numpy()
        boxes = detected_bboxes.xyxy.cpu().numpy()
        for cls,conf,box in zip(clss,confs,boxes):
            x1,y1,x2,y2 = box
            out_bboxes.append([x1,y1,x2,y2,conf])
        res = np.stack(out_bboxes,axis=0)
        return res
