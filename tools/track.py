'''
@File: track.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 29, 2025
@HomePage: https://github.com/YanJieWen
'''

import sys
sys.path.append('./external')

from loguru import logger
import torch

from utils import yaml_load, Config,hota_preprocess,get_hota_params,setup_logger
from data import build_dataloader
from modeling import build_detector
from engine.evaluator import Evaluator

import os
import shutil
import os.path as osp
from glob import glob
import emoji
import argparse

import motmetrics as mm
from pathlib import Path
from collections import OrderedDict
from TrackEval import trackeval
import pandas as pd


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info(f"{emoji.emojize(':rocket:')} Computing {k}...")
            #todo: iou阈值设置为0.5
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning(f'{emoji.emojize(":warning:")} No ground truth for {k}, skipping.')

    return accs, names

def main():
    parser = argparse.ArgumentParser('--Track main script--')
    parser.add_argument('-c','--config',type=str,default='configs/mm_crash_centernet.yaml',help='Config as yaml form')
    args = parser.parse_args()
    cfg_dict = yaml_load(args.config)
    main_cfg = Config(cfg_dict)
    track_cfg = main_cfg.Track
    data_cfg = track_cfg.dataset


    #step1: config setting
    file_name = osp.join('runs',main_cfg.task_name) #save_dir
    os.makedirs(file_name, exist_ok=True)
    results_folder = osp.join(file_name,'track_results')
    os.makedirs(results_folder, exist_ok=True)
    setup_logger(file_name, distributed_rank=0, filename=f'{main_cfg.task_name}.txt', mode='a')
    logger.info(f"{emoji.emojize(':clown_face:')*3}=>log can be found in {main_cfg.task_name}.txt")

    #step2:get dataloader
    val_loader = build_dataloader(data_cfg)
    evaluator = Evaluator(track_cfg,val_loader)
    logger.info(f"{emoji.emojize(':clown_face:')*3}=>Dataloader and Evaluator has been built!")
    #step3:get detector
    detector = build_detector(main_cfg)
    logger.info(f"{emoji.emojize(':clown_face:')*3}=>{track_cfg.model}-based detector has been built!")
    if len(os.listdir(results_folder))==0:
        if track_cfg.model=='openmm':
            *_,summary = evaluator.evaluate_mm(detector, result_folder=results_folder)
        elif track_cfg.model=='masort':
            *_,summary = evaluator.evaluate_ult(detector, result_folder=results_folder)
        logger.info('\n' + summary)
    #stepp4: evaluate mota
    mm.lap.default_solver = 'lap'
    if data_cfg.json_file == 'val_half.json':
        gt_type = '_val_half'
    else:
        gt_type = ''
    if main_cfg.benchmark_setting.name == 'mot20':
        gtfiles = glob(osp.join('./datasets/MOT20/train', f'*/gt/gt{gt_type}.txt'))
    elif main_cfg.benchmark_setting.name == 'mot17':
        gtfiles = glob(osp.join('./datasets/mot/train', f'*/gt/gt{gt_type}.txt'))
    elif main_cfg.benchmark_setting.name == 'dance':
        gtfiles = glob(osp.join('./datasets/dancetrack/val', f'*/gt/gt.txt'))
    elif main_cfg.benchmark_setting.name == 'Crash-Seq':
        gtfiles = glob(osp.join('./datasets/Crash-Seq/train', f'*/gt/gt{gt_type}.txt'))
    else:
        raise ValueError(f'{main_cfg.benchmark_setting.name} is not found!')
    tsfiles = [f for f in glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]
    logger.info(f"{emoji.emojize(':magnifying_glass_tilted_right:')}=>Found {len(gtfiles)} GTs and {len(tsfiles)} Preds")
    logger.info(f"{emoji.emojize(':hammer:')}=>Available LAP solver{mm.lap.available_solvers}")
    logger.info(f"{emoji.emojize(':hammer:')}=>Default LAP solver{mm.lap.available_solvers}")
    logger.info(f"{emoji.emojize(':envelope:' * 5)}==Loading files=={emoji.emojize(':envelope:' * 5)}")

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(Path(f).parts[-1].split('.')[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])
    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)
    logger.info(f"{emoji.emojize(':clown_face:' * 3)}==Running metrics=={emoji.emojize(':clown_face:' * 3)}")
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    logger.info(f"{emoji.emojize(':dog_face:')*3}=>Percentage version format")
    logger.info("\n" + mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))
    logger.info(f"{emoji.emojize(':dog_face:')*3}=>Original version format")
    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    logger.info("\n" + mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    logger.info(f"{emoji.emojize(':dog_face:')*3}{'='*10}Evaluate HOTA metrics{'='*10}")

    EVAL_CONFIG, METRCIS_CONFIG, DATASET_CONFIG = get_hota_params(main_cfg.benchmark_setting.name)
    root_name = DATASET_CONFIG['BENCHMARK'] + '-' + DATASET_CONFIG['SPLIT_TO_EVAL']
    hota_preprocess(DATASET_CONFIG, gtfiles, tsfiles)
    evaluator = trackeval.Evaluator(EVAL_CONFIG)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(DATASET_CONFIG)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in METRCIS_CONFIG['METRICS']:
            metrics_list.append(metric(METRCIS_CONFIG))
    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
    ago_name = Path(tsfiles[0]).parts[-2]
    output = output_res['MotChallenge2DBox'][ago_name]
    key_metrics_hota = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr']
    one_metric = {}
    for key, value in output.items():
        hot_rel = value['pedestrian']['HOTA']
        for m in key_metrics_hota:
            if m in hot_rel.keys():
                one_metric.setdefault(key, {}).setdefault(m, hot_rel[m].mean())
    df = pd.DataFrame(one_metric).transpose()
    df.index = df.index.map(lambda x: 'OVERALL' if x == 'COMBINED_SEQ' else x)
    df_hota = pd.merge(df, summary, how='right', left_index=True, right_index=True)
    hota_fmt = mh.formatters.copy()
    hota_metric = mm.io.motchallenge_metric_names.copy()
    update_fmt_dict = {}
    for k in key_metrics_hota:
        hota_fmt.setdefault(k.lower(), hota_fmt['mota'])
        hota_metric.setdefault(k.lower(), k)
    logger.info("\n" + mm.io.render_summary(df_hota, formatters=hota_fmt, namemap=hota_metric))
    # 将gt的文件夹名称回到原来的名称
    os.rename(os.path.join(DATASET_CONFIG['GT_FOLDER'], root_name),
              os.path.join(DATASET_CONFIG['GT_FOLDER'], DATASET_CONFIG['SPLIT_TO_EVAL']))


if __name__ == '__main__':
    main()