'''
@File: mm_test.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 23, 2025
@HomePage: https://github.com/YanJieWen
'''

import argparse
from loguru import logger
import time
from utils import yaml_load, setup_logger, logger_attrs
import os
from pathlib import Path
from glob import glob
import torch
import emoji
import mmcv
import json
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import build_dp, compat_cfg, get_device, replace_cfg_vals, setup_multi_processes, update_data_root


def main():
    parser = argparse.ArgumentParser('--mmdetection Testing--')
    parser.add_argument('--config', type=str, default='./configs/mm_mot.yaml', help='Config as yaml form')
    args = parser.parse_args()
    config = yaml_load(args.config)
    cfg = Config(config)
    log_root = os.path.join('runs', cfg.task_name)
    setup_logger(log_root, distributed_rank=0, filename='mmlog.txt', mode='a')
    test_cfg = cfg.detect.test
    logger_attrs(test_cfg)
    logger.info(f'{emoji.emojize(":rocket:")}=>{test_cfg.model}')
    train_work_dir = os.path.join('runs', cfg.task_name, cfg.detect.train.work_dir)
    assert os.path.exists(train_work_dir), 'Please run mm_train.py first'

    logger.info(f'{emoji.emojize(":clown_face:") * 3}=>modulation testing config')
    model_cfg = glob(os.path.join(train_work_dir, '*.py'))[0]
    model_cfg = Config.fromfile(model_cfg)
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
    test_dataloader_default_args = dict(samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **model_cfg.data.get('test_dataloader', {})
    }
    test_work_dir = os.path.join(Path(train_work_dir).parent, test_cfg.work_dir)
    os.makedirs(test_work_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    json_file = os.path.join(test_work_dir, f'eval_{timestamp}.json')
    dataset = build_dataset(model_cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    model_cfg.model.train_cfg = None
    model = build_detector(model_cfg.model, test_cfg=model_cfg.get('test_cfg'))
    fp16_cfg = model_cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    logger.info(f'{emoji.emojize(":clown_face:") * 3}=> Building Detector')
    checkpoint = load_checkpoint(model, os.path.join(train_work_dir, test_cfg.checkpoint), map_location='cpu')
    if test_cfg.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model = build_dp(model, model_cfg.device, device_ids=[0])
    logger.info(f'{emoji.emojize(":clown_face:") * 3}=>Begin Testing')
    outputs = single_gpu_test(model, data_loader, test_cfg.show, test_cfg.show_dir, test_cfg.show_score_thr)

    logger.info(f'{emoji.emojize(":clown_face:") * 3}=>Writing results&convert mmform into coco form')
    out_file = os.path.join(test_work_dir, test_cfg.out)
    anns = json.load(open(dataset.ann_file, 'r'))
    img_ids = [x['id'] for x in anns['images']]
    assert len(img_ids) == len(outputs), f'There are empty detections'
    ress = []
    for out_per_img, img_id in zip(outputs, img_ids):
        cls_ids = 1
        for out_per_img_per_cls in out_per_img:
            for out_per_img_per_cls_per_ins in out_per_img_per_cls:
                x1, y1, x2, y2, s = out_per_img_per_cls_per_ins
                res = {}
                res['image_id'] = img_id
                res['category_id'] = cls_ids
                res['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                res['score'] = float(s)
                ress.append(res)
            cls_ids += 1
    json.dump(ress, open(out_file, 'w'))
    eval_kwargs = model_cfg.get('evaluation', {}).copy()
    for key in [
        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        'rule', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox'))
    metric = dataset.evaluate(outputs, **eval_kwargs)
    logger.info(f'{emoji.emojize(":clown_face:") * 3}=>{metric}')
    metric_dict = dict(config=glob(os.path.join(train_work_dir, '*.py'))[0], metric=metric)
    mmcv.dump(metric_dict, json_file)
    logger.info(
        f'{emoji.emojize(":clown_face:") * 3}=>{os.path.basename(glob(os.path.join(train_work_dir, "*.py"))[0]).split(".")[0]} has been finished!')


if __name__ == '__main__':
    main()
