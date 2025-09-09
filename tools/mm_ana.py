'''
@File: mm_ana.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 23, 2025
@HomePage: https://github.com/YanJieWen
'''

import argparse
from loguru import logger
from utils import yaml_load, Config, setup_logger, logger_attrs, draw
import emoji
import os
from glob import glob
import cv2
import numpy as np
from PIL import Image

from mmdet.apis import init_detector, inference_detector


def main():
    parser = argparse.ArgumentParser('--mmdetection anaysis--')
    parser.add_argument('--config', type=str, default='./configs/mm_mot.yaml', help='Config as yaml form')
    args = parser.parse_args()
    config = yaml_load(args.config)
    cfg = Config(config)
    log_root = os.path.join('runs', cfg.task_name)
    setup_logger(log_root, distributed_rank=0, filename='mmlog.txt', mode='a')
    ana_cfg = cfg.detect.anaysis
    logger_attrs(ana_cfg)
    logger.info(f'{emoji.emojize(":rocket:")}=>{ana_cfg.model}')
    assert os.path.isfile(ana_cfg.demo_file), f'{ana_cfg.demo_file} is not exsits'

    out_root = os.path.join(log_root, 'anaysis')
    os.makedirs(out_root, exist_ok=True)
    model_config = glob(os.path.join(log_root, 'train', '*.py'))[0]
    ckpt_file = os.path.join(log_root, 'train', ana_cfg.ckpt)
    img = cv2.imread(ana_cfg.demo_file)
    logger.info(f'{emoji.emojize(":clown_face:") * 3}=>Building Detector')
    model = init_detector(config=model_config, checkpoint=ckpt_file, device='cuda:0')
    result = inference_detector(model=model, imgs=[img])
    cls_dict = {k: v for k, v in enumerate(model.CLASSES)}

    # convert result
    reshape_ress = []
    for img_res in result:
        reshape_res = []
        for c, res in enumerate(img_res):
            _res = np.insert(res, 5, int(c), axis=1)
            reshape_res.append(_res)
        reshape_res = np.concatenate(reshape_res)
        reshape_ress.append(reshape_res)
    res = reshape_ress[0]
    img = Image.fromarray(img[..., ::-1])
    draw(res, img, cls_dict,ana_cfg.draw_thresh)
    out_file = os.path.join(out_root, f'{os.path.basename(ana_cfg.demo_file)}')
    img.save(out_file)
    logger.info(f"{emoji.emojize(':clown_face:') * 3}=>demo has been saved at {out_file}")


if __name__ == '__main__':
    main()