'''
@File: demo.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 9月 04, 2025
@HomePage: https://github.com/YanJieWen
'''
import argparse

from loguru import logger
import emoji
import torch
import numpy as np
import cv2


from utils import yaml_load,Config,setup_logger
from modeling import build_detector
from modeling import build_tracker
from utils import xyxy2xywh,demo_openmm,demo_masort,dti
from engine.evaluator import write_results

from tqdm import tqdm
import os
import time

def main():
    parser = argparse.ArgumentParser('--Run Demo--')
    parser.add_argument('-c','--config',type=str,default='configs/ult_crash_cvmrs.yaml')
    parser.add_argument('-d','--demo',type=str,default='./demo/demo02.avi')
    args = parser.parse_args()
    cfg_dict = yaml_load(args.config)
    video_path = args.demo
    main_cfg = Config(cfg_dict)
    track_name = main_cfg.Track.model #决定启用何种模式处理输入的图像

    #step1: settings
    file_name = os.path.join('runs',main_cfg.task_name)
    os.makedirs(file_name,exist_ok=True)
    setup_logger(file_name,distributed_rank=0,filename=f'{main_cfg.task_name}.txt',mode='a')
    logger.info(f"{emoji.emojize(':clown_face:')*3}=>log can be found in {main_cfg.task_name}.txt")

    #setp2: get detector and tracker
    detector = build_detector(main_cfg)
    logger.info(f"{emoji.emojize(':clown_face:')*3}=={main_cfg.task_name} detector has been built==")
    tracker = build_tracker(main_cfg.Track)
    logger.info(f"{emoji.emojize(':clown_face:')*3}=={main_cfg.Track.model}-based Tracker has been built==")

    #step3: init video
    video_info = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file.")
        exit()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(5))
    print('fps:', fps)
    video_info['width'] = width
    video_info['height'] = height
    video_info['fps'] = fps

    demo_root = os.path.join(file_name,'demo')
    os.makedirs(demo_root,exist_ok=True)
    #step4:get_results
    if track_name=='openmm':
        results,inference_time,track_time = demo_openmm(cap,video_info,detector,tracker,main_cfg.Track,demo_root)
    elif track_name=='masort': #todo: masort-tracker
        results,inference_time,track_time = demo_masort(cap,video_info,detector,tracker,main_cfg.Track,demo_root)
    write_results(os.path.join(demo_root,'data.txt'),results)
    logger.info(f'Detection time:{inference_time}s\t Tracking time:{track_time}s')
    #step5: interpolate tracklets
    ori_path = os.path.join(demo_root, 'data.txt')
    if os.path.isfile(ori_path):
        out_path = os.path.join(demo_root,'data_dti.txt')
        dti(ori_path, out_path, n_min=25, n_dti=20)
        logger.info(f'Tracklets has been interpolation in {out_path}')
    else:
        logger.error(f'{ori_path} is not found')

if __name__ == '__main__':
    main()
