'''
@File: set_eval.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 29, 2025
@HomePage: https://github.com/YanJieWen
'''
from loguru import logger
import shutil


import os
from pathlib import Path
import emoji

def get_hota_params(data_type='mot17'):
    EVAL_CONFIG= {'PRINT_RESULTS':False,'RETURN_ON_ERROR':True,'DISPLAY_LESS_PROGRESS':False}
    METRCIS_CONFIG = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    if data_type=='mot17':
        DATASET_CONFIG = {'GT_FOLDER':'./datasets/mot/',
                                'TRACKERS_FOLDER': f'./My_outputs/{data_type}/',
                                'TRACKERS_TO_EVAL': None, #如果为None则会os.listdir
                                'TRACKER_SUB_FOLDER': 'data',
                                'BENCHMARK':'MOT17',
                                'SPLIT_TO_EVAL':'train',
                                'SKIP_SPLIT_FOL':False,
                                'GT_LOC_FORMAT':'{gt_folder}/{seq}/gt/gt_val_half.txt'}
    elif data_type=='mot20':
        DATASET_CONFIG = {'GT_FOLDER': './datasets/MOT20/',
                          'TRACKERS_FOLDER': f'./My_outputs/{data_type}/',
                          'TRACKERS_TO_EVAL': None,  # 如果为None则会os.listdir
                          'TRACKER_SUB_FOLDER': 'data',
                          'BENCHMARK': 'MOT20',
                          'SPLIT_TO_EVAL': 'train',
                          'SKIP_SPLIT_FOL': False,
                          'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt_val_half.txt'}
    elif data_type=='dance':
        DATASET_CONFIG = {'GT_FOLDER': './datasets/dancetrack/',
                          'TRACKERS_FOLDER': f'./My_outputs/{data_type}/',
                          'TRACKERS_TO_EVAL': None,  # 如果为None则会os.listdir
                          'TRACKER_SUB_FOLDER': 'data',
                          'BENCHMARK': 'Dancetrack',
                          'SPLIT_TO_EVAL': 'val',
                          'SKIP_SPLIT_FOL': False,
                          'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt'}
    elif data_type == 'Crash-Seq':
        DATASET_CONFIG = {'GT_FOLDER': './datasets/Crash-Seq/',
                          'TRACKERS_FOLDER': f'./My_outputs/{data_type}/',
                          'TRACKERS_TO_EVAL': None,  # 如果为None则会os.listdir
                          'TRACKER_SUB_FOLDER': 'data',
                          'BENCHMARK': 'Crash-Seq',
                          'SPLIT_TO_EVAL': 'train',
                          'SKIP_SPLIT_FOL': False,
                          'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt_val_half.txt'}
    else:
        raise ValueError(f'{data_type} is not support!')
    return EVAL_CONFIG,METRCIS_CONFIG,DATASET_CONFIG


def hota_preprocess(dataset_config,gtfiles,tsfiles):
    # step1：创建一个seqmaps存放seq名字
    seq_root = os.path.join(dataset_config['GT_FOLDER'], 'seqmaps')
    os.makedirs(seq_root, exist_ok=True)
    root_name = dataset_config['BENCHMARK'] + '-' + dataset_config['SPLIT_TO_EVAL']
    seq_names = ['name'] + [Path(x).parts[-3] for x in gtfiles]
    with open(os.path.join(seq_root, f'{root_name}.txt'), 'w') as f:
        for item in seq_names:
            f.write(f'{item}\n')
    logger.info(f"{emoji.emojize(':antenna_bars:')} \t Generate a seqmaps --> {os.path.join(seq_root, f'{root_name}.txt')} ")
    # step2: 创建tracker的文件夹
    split_root = Path(tsfiles[0]).parts
    ago_name = split_root[-2]
    track_fol = os.path.join(dataset_config['TRACKERS_FOLDER'], root_name, ago_name,
                             dataset_config['TRACKER_SUB_FOLDER'])
    os.makedirs(track_fol, exist_ok=True)
    _ = [shutil.copy(f, os.path.join(track_fol, os.path.basename(f))) for f in tsfiles if os.path.isfile(f)]
    logger.info(
        f"{emoji.emojize(':antenna_bars:')} \t Transfer track result from {os.path.dirname(tsfiles[0])} --> {track_fol} ")
    # step3:将GT的结果移到指定的文件夹下
    os.rename(os.path.join(dataset_config['GT_FOLDER'], dataset_config['SPLIT_TO_EVAL']), os.path.join(dataset_config['GT_FOLDER'], root_name))
    logger.info(f"{emoji.emojize(':antenna_bars:')} \t Rename GT folder from train --> {root_name} ")
