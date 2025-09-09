'''
@File: ult_train.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 30, 2025
@HomePage: https://github.com/YanJieWen
'''
import copy
import argparse
import os
import emoji
from loguru import logger

from utils import yaml_load,Config,setup_logger
from ultralytics import YOLO

import torch

def main():
    parser = argparse.ArgumentParser('--ultralytics training--')
    parser.add_argument('--config',type=str,default='./configs/crash/ultralytics_crash.yaml')
    args = parser.parse_args()
    cfg = yaml_load(args.config)
    cfg = Config(cfg)
    out_root = os.path.join('runs',cfg.task_name)
    os.makedirs(out_root,exist_ok=True)
    setup_logger(out_root, distributed_rank=0, filename=f'{cfg.task_name}.txt', mode='a')
    #step1: 初始化模型
    train_cfg = cfg.detect.train
    logger.info(f'=={train_cfg.__dict__}==')
    model = YOLO(train_cfg.pretrained)
    logger.info(f'=={emoji.emojize(":rocket:")*3}==Model has been built{emoji.emojize(":rocket:")*3}==')

    #step2: 将参数执行替换
    train_dict = copy.deepcopy(train_cfg).__dict__
    train_dict.pop('model')
    logger.info(f'=={emoji.emojize(":rocket:")*3}==Training has begun{emoji.emojize(":rocket:")*3}==')
    model.train(**train_dict)


if __name__ == '__main__':
    main()
