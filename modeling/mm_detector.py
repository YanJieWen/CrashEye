'''
@File: mm_detector.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 26, 2025
@HomePage: https://github.com/YanJieWen
'''
from copy import deepcopy
import torch
import numpy as np
from utils import yaml_load,Config
from mmcv import Config as mcfg
import os
import os.path as osp
import cv2
from glob import glob

from mmcv.ops import RoIPool
from mmdet.utils import build_dp, compat_cfg, get_device, replace_cfg_vals, setup_multi_processes, update_data_root
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor

class MMdetector():
    def __init__(self,cfg,task_name='mm_yolox-s_mot',work_dir='train'):
        if hasattr(cfg,'input_size'):
            cfg.input_size = eval(cfg.input_size)
        else:
            raise ValueError("Please supply input size in your yaml file")
        self.model_cfg_file = osp.join('runs',task_name,work_dir)
        self.det_cfg = cfg
        assert osp.exists(self.model_cfg_file), 'Please run mm_train.py first'
        self.model_cfg = self.convert_model_cfg()
        self.model = build_detector(self.model_cfg.model, test_cfg=self.model_cfg.get('test_cfg'))
        checkpoint = load_checkpoint(self.model, os.path.join(self.model_cfg_file, self.det_cfg.ckpt), map_location='cpu')
        self.model.CLASSES = checkpoint['meta']['CLASSES']
        self.model = build_dp(self.model, self.model_cfg.device, device_ids=[0])
        self.model.cfg = self.model_cfg
        self.model.to(self.model_cfg.device)
        self.model.eval()

    def convert_model_cfg(self):
        model_cfg = glob(osp.join(self.model_cfg_file, '*.py'))[0]
        model_cfg = mcfg.fromfile(model_cfg)
        model_cfg = replace_cfg_vals(model_cfg)
        update_data_root(model_cfg)
        model_cfg = compat_cfg(model_cfg)
        setup_multi_processes(model_cfg)
        if model_cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        if 'pretrained' in model_cfg.model:
            model_cfg.model.pretrained = None
        elif 'init_cfg' in model_cfg.model.backbone:
            model_cfg.model.backbone.init_cfg = None
        if model_cfg.model.get('neck'):
            if isinstance(model_cfg.model.neck, list):
                for neck_cfg in model_cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif model_cfg.model.neck.get('rfp_backbone'):
                if model_cfg.neck.rfp_backbone.get('pretrained'):
                    model_cfg.model.neck.rfp_backbone.pretrained = None
        model_cfg.device = get_device()
        model_cfg.model.train_cfg = None

        # model_cfg.model.input_size = self.det_cfg.input_size #不需要传入，被存储在model.data中
        model_cfg.model.test_cfg.score_thr = self.det_cfg.score_thr
        if hasattr(model_cfg.model.test_cfg,'nms'):
            model_cfg.model.test_cfg.nms.iou_threshold = self.det_cfg.nms_thresh
        elif hasattr(model_cfg.model.test_cfg,'rcnn'):
            model_cfg.model.test_cfg.rcnn.nms.iou_threshold = self.det_cfg.nms_thresh
        else:
            pass
        return model_cfg
    def get_img_metas(self,img):
        if len(img.shape) == 4:
            img = img[0]
            if isinstance(img, torch.Tensor):
                img = np.asarray(img.cpu().numpy().transpose(1, 2, 0) * 255., dtype=np.uint8)[..., ::-1]  # rgb->bgr
        imgs = [img]
        is_batch = False
        _cfg = self.model.cfg
        cfg_meta = _cfg.copy()
        cfg_meta.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        cfg_meta.data.test.pipeline = replace_ImageToTensor(cfg_meta.data.test.pipeline)
        test_pipeline = Compose(cfg_meta.data.test.pipeline)
        datas = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                data = dict(img=img)
            else:
                data = dict(img_info=dict(filename=img), img_prefix=None)
            data = test_pipeline(data)
            datas.append(data)
        data = collate(datas, samples_per_gpu=len(imgs))
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        device = next(self.model.parameters()).device
        if next(self.model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'
        return data
    def __call__(self, img): #tensor->(b,c,h,w)/255 and  to bgr
        '''
        逐帧输出检测结果
        Args:
            img: torch.Tensor-->(b,c,h,w)-->经过归一化且为RGB图像
            需要在img_metas中转为3元组，*255 uint8且为BGR格式，mmdet内部会自动转为RGB图像
        Returns: np.array-->(m,5)-->(x1,y1,x2,y2,conf)

        '''
        data = self.get_img_metas(img)
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)
        res = results[0][0]  # 第一个0是batch，第二个0是cls的索引
        return res





# if __name__ == '__main__':
#     from data import build_dataloader
#     config = yaml_load('./configs/mm_mot.yaml')
#
#
#     cfg = Config(config)
#     data_cfg = cfg.Track.dataset
#     det_cfg = cfg.Track.detector
#     val_loader = build_dataloader(data_cfg)
#     detector = MMdetector(det_cfg,cfg.task_name,cfg.detect.train.work_dir)
#     img, _, info_imgs, ids = next(iter(val_loader))
#     print(detector(img))

    # det_cfg.input_size = eval(det_cfg.input_size)
    #
    # val_loader = build_dataloader(data_cfg)
    # train_work_dir = osp.join('runs',cfg.task_name,cfg.detect.train.work_dir)
    # assert os.path.exists(train_work_dir), 'Please run mm_train.py first'
    # #step1: convert model_cfg
    # model_cfg = glob(osp.join(train_work_dir,'*.py'))[0]
    # model_cfg = mcfg.fromfile(model_cfg)
    # model_cfg = replace_cfg_vals(model_cfg)
    # update_data_root(model_cfg)
    # model_cfg = compat_cfg(model_cfg)
    # setup_multi_processes(model_cfg)
    # if model_cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True
    # if 'pretrained' in model_cfg.model:
    #     model_cfg.model.pretrained = None
    # elif 'init_cfg' in model_cfg.model.backbone:
    #     model_cfg.model.backbone.init_cfg = None
    # if model_cfg.model.get('neck'):
    #     if isinstance(model_cfg.model.neck, list):
    #         for neck_cfg in model_cfg.model.neck:
    #             if neck_cfg.get('rfp_backbone'):
    #                 if neck_cfg.rfp_backbone.get('pretrained'):
    #                     neck_cfg.rfp_backbone.pretrained = None
    #     elif model_cfg.model.neck.get('rfp_backbone'):
    #         if model_cfg.neck.rfp_backbone.get('pretrained'):
    #             model_cfg.model.neck.rfp_backbone.pretrained = None
    # model_cfg.device = get_device()
    # model_cfg.model.train_cfg = None
    # model_cfg.model.input_size = det_cfg.input_size
    # model_cfg.model.test_cfg.score_thr = det_cfg.score_thr
    # model_cfg.model.test_cfg.nms.iou_threshold = det_cfg.nms_thresh
    # #step2: 实例化模型
    # model = build_detector(model_cfg.model,test_cfg=model_cfg.get('test_cfg'))
    # checkpoint = load_checkpoint(model, os.path.join(train_work_dir, det_cfg.ckpt), map_location='cpu')
    # model.CLASSES = checkpoint['meta']['CLASSES']
    # model = build_dp(model, model_cfg.device, device_ids=[0])
    # model.cfg = model_cfg
    # model.to(model_cfg.device)
    # model.eval()
    #
    # #step3: generate img_metas
    # img, _, info_imgs, ids = next(iter(val_loader))
    # if len(img.shape) == 4:
    #     img = img[0]
    #     if isinstance(img, torch.Tensor):
    #         img = np.asarray(img.numpy().transpose(1, 2, 0)*255., dtype=np.uint8)[..., ::-1]  # rgb->bgr
    # imgs = [img]
    # is_batch = False
    # _cfg = model.cfg
    # cfg_meta = _cfg.copy()
    # cfg_meta.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    # cfg_meta.data.test.pipeline = replace_ImageToTensor(cfg_meta.data.test.pipeline)
    # test_pipeline = Compose(cfg_meta.data.test.pipeline)
    # datas = []
    # for img in imgs:
    #     if isinstance(img, np.ndarray):
    #         data = dict(img=img)
    #     else:
    #         data = dict(img_info=dict(filename=img), img_prefix=None)
    #     data = test_pipeline(data)
    #     datas.append(data)
    # data = collate(datas, samples_per_gpu=len(imgs))
    # data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    # data['img'] = [img.data[0] for img in data['img']]
    # device = next(model.parameters()).device
    # if next(model.parameters()).is_cuda:
    #     data = scatter(data, [device])[0]
    # else:
    #     for m in model.modules():
    #         assert not isinstance(
    #             m, RoIPool
    #         ), 'CPU inference with RoIPool is not supported currently.'
    # #step5: begin infer
    # with torch.no_grad():
    #     results = model(return_loss=False, rescale=True, **data)
    # res = results[0][0] #第一个0是batch，第二个0是cls的索引
    # print(res[:,-1].min())



