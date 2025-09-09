'''
@File: mm_tracker.py
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
from mmtrack.models import build_model
import mmcv

from data import build_dataloader
from utils import yaml_load,Config
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
#确保注册的mmtrack中的pipeline

import mmtrack.datasets.pipelines



class MMTracker():

    def __init__(self,track_cfg):
        self.cfg = track_cfg
        self.config = mmcv.Config.fromfile(self.cfg.config)
        self.transform_config_()
        self.model = build_model(self.config.model)
        self.model.init_weights()
        self.model.cfg = self.config
        self.model.to("cuda:0")
        self.model.eval()
    @staticmethod
    def convert_tensor_to_array(img):
        if isinstance(img,torch.Tensor):
            img = img[0]
            img = np.asarray(img.cpu().numpy().transpose(1, 2, 0) * 255., dtype=np.uint8)[..., ::-1]
        return img
    def transform_config_(self):
        self.config.model.detector.pretrained = None
        update_dict = self.cfg.__dict__
        for k, v in update_dict.items():
            if hasattr(self.config.model.tracker, k):
                self.config.model.tracker[k] = v
    def get_img_metas(self,img,frame_id):
        _cfg = self.model.cfg
        device = next(self.model.parameters()).device
        data = dict(img=img, img_info=dict(frame_id=frame_id), img_prefix=None)
        _cfg = _cfg.copy()
        _cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        _cfg.data.test.pipeline[1].img_scale = eval(self.cfg.img_size)
        test_pipeline = Compose(_cfg.data.test.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device])[0]
        img_metas = data['img_metas'][0]
        return data,img_metas
    @staticmethod
    def convert_output_format(track_bboxes,track_ids):
        track_bboxes = deepcopy(track_bboxes)
        track_bboxes = track_bboxes.detach().cpu().numpy()
        track_ids = track_ids.detach().cpu().numpy()
        track_bboxes[:, 2:4] -= track_bboxes[:, :2]  # xyxy->tlwh
        res = np.hstack([track_bboxes, track_ids.reshape(-1, 1)])
        return res

    def __call__(self,output_results,img,frame_id):
        '''

        Args:
            output_results: ndarray->(N,5)->[x1,y1,x2,y2,conf]
            img: torch.tensor->(1,3,h,w) with RGB->(H,W,3) with BGR array
            frame_id:int=0,1,2,...

        Returns: array[x1,y1,w,h,conf,pid]

        '''
        #todo: 判断输出边界框
        # a = deepcopy(output_results)
        # a[:,2:4] -= a[:,:2]
        # print(a[:,2:4])

        img = self.convert_tensor_to_array(img)
        data, img_metas = self.get_img_metas(img,frame_id)
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 1:
            self.model.tracker.reset()
        bbox_xyxy = []
        labels = []
        if len(output_results):
            for x1, y1, x2, y2, conf in output_results:
                obj = [x1, y1, x2, y2, conf]
                bbox_xyxy.append(obj)
                labels.append(0)
        det_bboxes = torch.Tensor(bbox_xyxy).detach().to('cpu')
        det_labels = torch.Tensor(labels).detach().to('cpu')
        # track_bboxes, track_labels, track_ids
        with torch.no_grad():
            track_bboxes, _, track_ids = self.model.tracker.track(
                img=data['img'][0],  # 这里的img已经经过了缩放
                img_metas=img_metas,
                model=self.model,
                bboxes=det_bboxes,
                labels=det_labels,
                frame_id=frame_id,
                rescale=True)
        outputs = self.convert_output_format(track_bboxes,track_ids)
        # print(track_ids)
        return outputs



# if __name__ == '__main__':
#     # 生成检测结果
#     frame_id = 0
#     det_res_x1y1 = np.random.uniform(32, 54, size=(32, 2))
#     det_res_x2y2 = np.random.uniform(60, 200, size=(32, 2))
#     det_res = np.hstack([det_res_x1y1, det_res_x2y2])
#     det_res = np.insert(det_res, 4, 1, axis=1)
#     #实例化
#     all_config = yaml_load('./configs/mm_mot.yaml')
#     cfg = Config(all_config)
#     track_cfg = cfg.Track.tracker
#     data_cfg = cfg.Track.dataset
#     val_loader = build_dataloader(data_cfg)
#     tracker = MMTracker(track_cfg)
#     #获取图像
#     img, _, info_imgs, ids = next(iter(val_loader))
#     res = tracker(det_res,img,frame_id)
#     print(res)

    # all_config = yaml_load('./configs/mm_mot.yaml')
    # cfg = Config(all_config)
    # track_cfg = cfg.Track.tracker
    # def convert_tensor_to_array(img):
    #     if isinstance(img, torch.Tensor):
    #         img = img[0]
    #         img = np.asarray(img.numpy().transpose(1, 2, 0) * 255., dtype=np.uint8)[..., ::-1]
    #     return img
    # data_cfg = cfg.Track.dataset
    # val_loader = build_dataloader(data_cfg)
    # img, _, info_imgs, ids = next(iter(val_loader))
    # img = convert_tensor_to_array(img)
    # frame_id = 0
    #
    # config_file = './modeling/mmtracking/configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
    # config = mmcv.Config.fromfile(config_file)
    # # det_config = mmcv.Config.fromfile()
    # #todo: 修改config参数
    # config.model.detector.pretrained = None
    # update_dict = track_cfg.__dict__
    # for k,v in update_dict.items():
    #     if hasattr(config.model.tracker,k):
    #         config.model.tracker[k] = v
    # #实例化模型
    # model = build_model(config.model)
    # model.init_weights()
    # model.cfg = config
    # model.to("cuda:0")
    # model.eval()
    # #构造img_metas
    # _cfg = model.cfg
    # device = next(model.parameters()).device
    # data = dict(img=img, img_info=dict(frame_id=frame_id), img_prefix=None)
    # _cfg = _cfg.copy()
    # _cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    # _cfg.data.test.pipeline[1].img_scale = (800,1440)
    # test_pipeline = Compose(_cfg.data.test.pipeline)
    # print(test_pipeline)
    # data = test_pipeline(data)
    # data = collate([data], samples_per_gpu=1)
    # data = scatter(data, [device])[0]
    # img_metas = data['img_metas'][0]
    # print(img_metas)
    # #
    # #生成一堆检测结果call
    # frame_id = img_metas[0].get('frame_id', -1)
    # if frame_id == 0:
    #     model.tracker.reset()
    # det_res_x1y1 = np.random.uniform(32, 54, size=(32,2))
    # det_res_x2y2 = np.random.uniform(60, 200, size=(32,2))
    # det_res = np.hstack([det_res_x1y1,det_res_x2y2])
    # det_res = np.insert(det_res,4,1,axis=1)
    #
    # bbox_xyxy = []
    # labels = []
    # if len(det_res):
    #     for x1, y1, x2, y2, conf in det_res:
    #         obj = [x1, y1, x2, y2, conf]
    #         bbox_xyxy.append(obj)
    #         labels.append(0)
    # det_bboxes = torch.Tensor(bbox_xyxy).to('cuda:0')
    # det_labels = torch.Tensor(labels).to('cuda:0')
    # #track_bboxes, track_labels, track_ids
    # # print(data['img'][0].shape)
    # track_bboxes, _, track_ids = model.tracker.track(
    #                 img=data['img'][0], #这里的img已经经过了缩放
    #                 img_metas=img_metas,
    #                 model=model,
    #                 bboxes=det_bboxes,
    #                 labels=det_labels,
    #                 frame_id=frame_id,
    #                 rescale=True)
    # #将结果转为评估的格式-->[t,k,w,h,score,pid]
    # track_bboxes = track_bboxes.detach().cpu().numpy()
    # track_ids = track_ids.detach().cpu().numpy()
    # track_bboxes[:,2:4] -= track_bboxes[:,:2] #xyxy->tlwh
    # res = np.hstack([track_bboxes,track_ids.reshape(-1,1)])
    # print(res.shape)
