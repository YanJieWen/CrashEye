'''
@File: mm_train.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 23, 2025
@HomePage: https://github.com/YanJieWen
'''

import warnings
import copy
import time
import argparse
import os
import emoji

from utils import yaml_load
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes,replace_cfg_vals,update_data_root,collect_env,get_device,get_root_logger
from mmcv import Config
from mmcv.utils import get_git_hash
import torch.distributed as dist

import torch

def main():
    parser = argparse.ArgumentParser('--mmdetection Training--')
    parser.add_argument('--config',type=str,default='./configs/mm_mot.yaml',help='Config as yaml form')
    args = parser.parse_args()
    cfg_dict = yaml_load(args.config)
    cfg = Config(cfg_dict)
    out_root = os.path.join('runs',cfg.task_name)
    os.makedirs(out_root,exist_ok=True)

    if not cfg.benchmark_setting.test:
        train_cfg = cfg.detect.train
    else:
        raise ValueError(f'Training has been finished, run test...')
    #step1: 配置cfg
    _cfg = Config.fromfile(train_cfg.config)
    _cfg = replace_cfg_vals(_cfg)
    update_data_root(_cfg)

    if train_cfg.auto_scale_lr:
        if 'auto_scale_lr' in _cfg and 'enable' in _cfg.auto_scale_lr and 'base_batch_size' in _cfg.auto_scale_lr:
            _cfg.auto_scale_lr.enable = True
    else:
        warnings.warn('Can not find "auto_scale_lr" or '
                            '"auto_scale_lr.enable" or '
                            '"auto_scale_lr.base_batch_size" in your'
                            ' configuration file. Please update all the '
                            'configuration files to mmdet >= 2.24.1.')
    setup_multi_processes(_cfg)
    if _cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    _cfg.work_dir = os.path.join(out_root,train_cfg.work_dir)
    _cfg.resume_from = train_cfg.resume_from
    _cfg.auto_resume = train_cfg.auto_resume
    if train_cfg.gpus is not None:
        _cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                        'single GPU mode in non-distributed training. '
                        'Use `gpus=1` now.')
    if train_cfg.gpu_ids is not None:
        _cfg.gpu_ids = train_cfg.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                        'Because we only support single GPU mode in '
                        'non-distributed training. Use the first GPU '
                        'in `gpu_ids` now.')

    if train_cfg.gpus is None and train_cfg.gpu_ids is None and torch.cuda.device_count():
        _cfg.gpu_ids = [train_cfg.gpu_id]
    os.makedirs(os.path.join(out_root,train_cfg.work_dir),exist_ok=True)
    _cfg.dump(os.path.join(_cfg.work_dir,os.path.basename(train_cfg.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(os.path.join(out_root,train_cfg.work_dir),f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=_cfg.log_level)
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
    meta['env_info'] = env_info
    meta['config'] = _cfg.pretty_text
    logger.info(f'Config:\n{_cfg.pretty_text}')
    _cfg.device = get_device()
    seed = init_random_seed(train_cfg.seed,device=_cfg.device)
    seed = seed + dist.get_rank() if train_cfg.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {train_cfg.deterministic}')
    _cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = os.path.basename(train_cfg.config)

    model = build_detector(
    _cfg.model,train_cfg=_cfg.get('train_cfg'),
    test_cfg=_cfg.get('test_cfg')
    )
    model.init_weights()
    logger.info(f'{emoji.emojize(":clown_face:")*3}=>{os.path.basename(train_cfg.config).split(".")[0]} has been initialized!')

    datasets = [build_dataset(_cfg.data.train)]
    logger.info(f'{emoji.emojize(":rocket:")*3}=>dataset has been initialized!')


    if len(_cfg.workflow)==2:
        assert 'val' in [mode for (mode, _) in _cfg.workflow]
        val_dataset = copy.deepcopy(_cfg.data.val)
        val_dataset.pipeline = _cfg.data.train.get(
                'pipeline', _cfg.data.train.dataset.get('pipeline'))
        datasets.append(build_dataset(val_dataset))
    if _cfg.checkpoint_config is not None:
        _cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
    logger.info(f'{emoji.emojize(":robot:")*3}=>Training begin')
    train_detector(model,datasets,_cfg,distributed=False,timestamp=timestamp,validate=not train_cfg.no_validate,meta=meta)

if __name__ == '__main__':
    main()
