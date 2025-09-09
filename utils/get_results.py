'''
@File: get_results.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 9月 05, 2025
@HomePage: https://github.com/YanJieWen
'''
import numpy as np
from loguru import logger
import time
import os
import cv2
from PIL import Image

import torch
from torch import distributed as dist

from utils import draw,draw_traj



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
def demo_openmm(cap,video_info,detector,tracker,track_cfg,demo_root):
    #todo:实现在线的轨迹可视化
    video_name = 'demo_openmm.avi'
    video_name = os.path.join(demo_root,video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, video_info['fps'], (video_info['width'], video_info['height']))
    #初始化读取
    object_trails = {}
    ids_dict = {}
    frame_id = 1
    inference_time = 0
    track_time = 0
    results = []
    while True:
        _, im = cap.read() #h,w,3 BGR
        if im is None:
            break
        img = Image.fromarray(im[...,::-1])
        start = time.time()
        bboxes = detector(im)
        infer_end = time_synchronized()
        inference_time += infer_end-start
        if len(bboxes)!=0:
            online_targets = tracker(bboxes,im,frame_id)
            online_tlwhs = []
            online_ids = []
            online_scores = []
        for x in online_targets:
            tlwh = x[:4]
            tid = x[5]
            vertical = tlwh[2] / tlwh[3]>track_cfg.aspect_ratio
            if tlwh[2] * tlwh[3] > track_cfg.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(x[4])
                #将中心点加入到object_trails以及tid加入到ids_dict
                xc, yc = tlwh[:2]+tlwh[2:] / 2
                if tid not in object_trails.keys():
                    object_trails.setdefault(tid,[]).append((int(xc),int(yc)))
                else:
                    object_trails[tid].append((int(xc),int(yc)))
                if tid not in ids_dict.keys():
                    ids_dict[tid] = f'ID{int(tid)}'

        if len(online_tlwhs)!=0:
            bboxes = np.stack(online_tlwhs,axis=0)
            bboxes[:,2:4] += bboxes[:,:2]
            res = np.zeros((len(online_tlwhs),6))
            res[:,:4] = bboxes
            res[:,4] = online_scores
            res[:,5] = online_ids
            draw(res,img,ids_dict)
            #todo:图像上绘制轨迹
            vis_trajs = {}
            for key,value in object_trails.items():
                if key in online_ids:
                    vis_trajs[key] = value
            # 移除不在当前帧中的轨迹
            # for id in object_trails.keys():
            #     if id not in online_ids:
            #         object_trails.pop(id)
            draw_traj(img,vis_trajs)
        img = np.array(img, dtype=np.uint8)[..., ::-1]
        video.write(img)
        results.append((frame_id, online_tlwhs, online_ids, online_scores))
        track_end = time_synchronized()
        track_time += track_end-infer_end
        frame_id += 1
    video.release()
    cv2.destroyAllWindows()
    synchronize()
    return results,inference_time,track_time

def demo_masort(cap,video_info,detector,tracker,track_cfg,demo_root):
    #todo:实现在线的轨迹可视化
    video_name = 'demo_masort.avi'
    video_name = os.path.join(demo_root,video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, video_info['fps'], (video_info['width'], video_info['height']))
    #初始化读取
    object_trails = {}
    ids_dict = {}
    frame_id = 1
    inference_time = 0
    track_time = 0
    results = []
    imgsz = track_cfg.dataset.img_size
    while True:
        _, im = cap.read() #h,w,3 BGR
        if im is None:
            break
        img = Image.fromarray(im[...,::-1])#Image(RGB)
        #处理图像img_tensor,img_info(h,w,fid,vid,img_file),
        res_im = transform_img(im,imgsz) #H,W,3-->bgr--uint8
        img_info = (im.shape[0],im.shape[1],frame_id,0,[im])
        start = time.time()
        bboxes = detector(res_im)
        infer_end = time_synchronized()
        inference_time += infer_end-start
        if len(bboxes)!=0:
            #todo:添加masort中的demo的处理
            online_targets = tracker.update(bboxes,img_info,imgsz)
            online_tlwhs = []
            online_ids = []
            online_scores = []
        for x in online_targets:
            tlwh = x.tlwh
            tid = x.track_id
            vertical = tlwh[2] / tlwh[3]>track_cfg.aspect_ratio
            if tlwh[2] * tlwh[3] > track_cfg.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(x.score)
                #将中心点加入到object_trails以及tid加入到ids_dict
                xc, yc = tlwh[:2]+tlwh[2:] / 2
                if tid not in object_trails.keys():
                    object_trails.setdefault(tid,[]).append((int(xc),int(yc)))
                else:
                    object_trails[tid].append((int(xc),int(yc)))
                if tid not in ids_dict.keys():
                    ids_dict[tid] = f'ID{int(tid)}'

        if len(online_tlwhs)!=0:
            bboxes = np.stack(online_tlwhs,axis=0)
            bboxes[:,2:4] += bboxes[:,:2]
            res = np.zeros((len(online_tlwhs),6))
            res[:,:4] = bboxes
            res[:,4] = online_scores
            res[:,5] = online_ids
            draw(res,img,ids_dict)
            #todo:图像上绘制轨迹
            vis_trajs = {}
            for key,value in object_trails.items():
                if key in online_ids:
                    vis_trajs[key] = value
            # 移除不在当前帧中的轨迹
            # for id in object_trails.keys():
            #     if id not in online_ids:
            #         object_trails.pop(id)

            draw_traj(img,vis_trajs)

        img = np.array(img, dtype=np.uint8)[..., ::-1]
        video.write(img)
        results.append((frame_id, online_tlwhs, online_ids, online_scores))
        track_end = time_synchronized()
        track_time += track_end-infer_end
        frame_id += 1
    video.release()
    cv2.destroyAllWindows()
    synchronize()
    return results,inference_time,track_time


def transform_img(img,imgsz):
    '''

    Args:
        img: BGF-->[H,W,3]
        imgsz: (H,W)

    Returns: H,W,C-->bgr--uint8

    '''
    padded_img = np.ones((imgsz[0],imgsz[1],3))*114.
    r = min(imgsz[0]/img.shape[0],imgsz[1]/img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # padded_img = padded_img[:, :, ::-1]
    padded_img = np.ascontiguousarray(padded_img, dtype=np.uint8)
    return padded_img