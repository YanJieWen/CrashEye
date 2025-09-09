'''
@File: ult_val.py
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

import inspect

def main():
    parser = argparse.ArgumentParser('--ultralytics testing--')
    parser.add_argument('--config', type=str, default='./configs/ultralytics_mot.yaml')
    args = parser.parse_args()
    cfg = yaml_load(args.config)
    cfg = Config(cfg)
    out_root = os.path.join('runs', cfg.task_name)
    os.makedirs(out_root, exist_ok=True)
    setup_logger(out_root, distributed_rank=0, filename=f'{cfg.task_name}.txt', mode='a')
    #step1:初始化模型
    test_cfg = cfg.detect.test
    logger.info(f'=={test_cfg.__dict__}==')
    model = YOLO(test_cfg.weight)
    logger.info(f'=={emoji.emojize(":rocket:")*3}Model has been built{emoji.emojize(":rocket:")*3}==')

    #step2: 评估模型
    test_dict = copy.deepcopy(test_cfg.__dict__)
    test_dict.pop('model')
    test_dict.pop('weight')

    metrics = model.val(**test_dict)
    iou_map = [metrics.box.map,metrics.box.map50,metrics.box.map75]
    metric_info = ','.join([
        "{}:{:.2f}".format(k,v) for k,v in zip(
            ["AP","AP50","AP75"],iou_map
        )
    ])
    info = metric_info + "\n"
    logger.info(info)

if __name__ == '__main__':
    main()
