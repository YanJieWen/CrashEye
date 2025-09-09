'''
@File: evaluator.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 28, 2025
@HomePage: https://github.com/YanJieWen
'''
from copy import deepcopy
from collections import defaultdict

import numpy as np
from tqdm import tqdm
# from loguru import logger

import torch
from torch import distributed as dist

import contextlib
import io
import os
import itertools
import json
import tempfile
import time

from modeling import build_tracker
from utils import xyxy2xywh

#cfg.Track
def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    # logger.info('save results to {}'.format(filename))
class Evaluator:
    def __init__(self,cfg,dataloader):
        self.img_size = cfg.dataset.img_size
        self.dataloader = dataloader
        self.cfg = cfg #track_cfg

    def evaluate_mm(self,detector,half=False,trt_file=None,decoder=None,
                    test_size=None,result_folder=None):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        if half:
            detector = detector.model.half()
        ids = [] #todo: 好像并未用到
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader)-1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            detector.model(x)
            model = model_trt
        tracker = build_tracker(self.cfg) #Frame-by-Frame

        #begin tracker
        for cur_iter, (imgs,_,info_imgs,ids) in enumerate(progress_bar(self.dataloader)):
            frame_id = info_imgs[2].item()
            video_id = info_imgs[3].item()
            img_file_name = info_imgs[4]
            video_name = img_file_name[0].split('/')[0]

            if video_name not in video_names:
                video_names[video_id] = video_name
            if frame_id==1:
                tracker = build_tracker(self.cfg)
                if len(results)!=0:
                    result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                    write_results(result_filename, results)
                    results = []
            imgs = imgs.type(tensor_type)
            is_time_record = cur_iter < len(self.dataloader) - 1
            if is_time_record:
                start = time.time()

            outputs = detector(imgs) #(m,5)-->[x1,y2,x2,y2,conf]

            if decoder is not None:
                outputs = decoder(outputs, dtype=outputs.type())
            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start
            #将结果转为原图尺寸并转为coco格式
            output_results = self.convert_to_coco_format(outputs,info_imgs,ids)
            data_list.extend(output_results)
            if len(outputs)!=0:
                online_targets = tracker(outputs,imgs,frame_id)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                #todo: mmshould transform original scale
                self.transform_rescale_(online_targets,info_imgs)
                for x in online_targets:
                    tlwh = x[:4]
                    tid = x[5]
                    vertical = tlwh[2] / tlwh[3] > self.cfg.aspect_ratio
                    if tlwh[2] * tlwh[3] > self.cfg.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(x[4])
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        eval_results = self.evaluate_detections(data_list, statistics)
        synchronize()
        return eval_results
    def evaluate_ult(self,detector,half=False,trt_file=None,decoder=None,
                    test_size=None,result_folder=None):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        if half:
            detector = detector.model.half()
        ids = [] #todo: 好像并未用到
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader)-1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            detector.model(x)
            model = model_trt
        tracker = build_tracker(self.cfg) #Frame-by-Frame
        ori_thresh = self.cfg.tracker.track_thresh

        #begin tracker
        for cur_iter, (imgs,_,info_imgs,ids) in enumerate(progress_bar(self.dataloader)):
            frame_id = info_imgs[2].item()
            video_id = info_imgs[3].item()
            img_file_name = info_imgs[4]
            video_name = img_file_name[0].split('/')[0]
            if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                self.cfg.tracker.track_buffer = 14
            elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                self.cfg.tracker.track_buffer = 25
            else:
                self.cfg.tracker.track_buffer = 30

            if video_name == 'MOT17-01-FRCNN':
                self.cfg.tracker.track_thresh = 0.65
            elif video_name == 'MOT17-06-FRCNN':
                self.cfg.tracker.track_thresh = 0.65
            elif video_name == 'MOT17-12-FRCNN':
                self.cfg.tracker.track_thresh = 0.7
            elif video_name == 'MOT17-14-FRCNN':
                self.cfg.tracker.track_thresh = 0.67
            elif video_name in ['MOT20-06', 'MOT20-08']:
                self.cfg.tracker.track_thresh = 0.3
            else:
                self.cfg.tracker.track_thresh = ori_thresh

            if video_name not in video_names:
                video_names[video_id] = video_name

            if frame_id==1:
                tracker = build_tracker(self.cfg)
                if len(results)!=0:
                    result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                    write_results(result_filename, results)
                    results = []
            imgs = imgs.type(tensor_type)
            is_time_record = cur_iter < len(self.dataloader) - 1
            if is_time_record:
                start = time.time()

            outputs = detector(imgs) #(m,5)-->[x1,y2,x2,y2,conf]

            if decoder is not None:
                outputs = decoder(outputs, dtype=outputs.type())
            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start
            #将结果转为原图尺寸并转为coco格式
            output_results = self.convert_to_coco_format(outputs,info_imgs,ids)
            data_list.extend(output_results)
            if len(outputs)!=0:
                online_targets = tracker.update(outputs,info_imgs,self.cfg.dataset.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > self.cfg.aspect_ratio
                    if tlwh[2] * tlwh[3] > self.cfg.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                results.append((frame_id, online_tlwhs, online_ids, online_scores))
            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        eval_results = self.evaluate_detections(data_list, statistics)
        synchronize()
        return eval_results
    def transform_rescale_(self,outputs,info_imgs):
        img_h = info_imgs[0].item()
        img_w = info_imgs[1].item()
        scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
        outputs[:,:4]/=scale
    def convert_to_coco_format(self,outputs,info_imgs,ids):
        outputs = deepcopy(outputs)
        data_list = []
        if isinstance(outputs,np.ndarray):
            outputs = [outputs]
        for (output,img_h,img_w,img_id) in zip(outputs,info_imgs[0],info_imgs[1],ids):
            if isinstance(img_h,torch.Tensor):
                img_h = img_h.item()
            if isinstance(img_w,torch.Tensor):
                img_w = img_w.item()
            if isinstance(img_id,torch.Tensor):
                img_id = img_id
            if output is None:
                continue
            if isinstance(output,torch.Tensor):
                output = output.cpu().numpy()
            bboxes = output[:, 0:4]
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)
            if output.shape[1]==5:
                cls = np.array([0]*len(output),dtype=int)
                scores = output[:,4]
            else:
                cls = output[:,6]
                scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                x1,x2,w,h = bboxes[ind]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [float(x1),float(x2),float(w),float(h)],
                    "score": float(scores[ind]),
                    "segmentation": [],
                }
                data_list.append(pred_data)
        return data_list
    def evaluate_detections(self,data_dict,statistics):
        # logger.info("Evaluate in main process...")
        annType = ["segm", "bbox", "keypoints"]
        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)
        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                ["forward", "track", "inference"],
                [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
            )
            ]
        )
        info = time_info + "\n"
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            from pycocotools.cocoeval import COCOeval
            coco_eval = COCOeval(cocoGt,cocoDt,annType[1])
            coco_eval.evaluate()
            coco_eval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            info += redirect_string.getvalue()
            return coco_eval.stats[0], coco_eval.stats[1], info
        else:
            return 0, 0, info

# if __name__ == '__main__':
#     from utils import yaml_load, Config
#     from data import build_dataloader
#     from modeling import build_detector
#
#     conf_file = './configs/mm_mot.yaml'
#     cfg = yaml_load(conf_file)
#     cfg = Config(cfg)
#     track_cfg = cfg.Track
#     result_folder = os.path.join('runs',cfg.task_name,'my_tracker',cfg.benchmark_setting.name)
#     os.makedirs(result_folder,exist_ok=True)
#
#     data_cfg = track_cfg.dataset
#     val_loader = build_dataloader(data_cfg)
#
#     eval = Evaluator(track_cfg,val_loader)
#     #输入参数
#     detector = build_detector(cfg)
#     res = eval.evaluate_mm(detector,result_folder=result_folder)
#     print(res)