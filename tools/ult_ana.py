'''
@File: ult_ana.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 30, 2025
@HomePage: https://github.com/YanJieWen
'''
from loguru import logger

import copy
import argparse
import os
import emoji
import json
from tqdm import tqdm

import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pandas as pd

from utils import yaml_load,Config,setup_logger
from ultralytics import YOLO

from modeling.CVMR.tools.get_heat import Yolov8_heatmap


def main():
    parser = argparse.ArgumentParser('--ultralytics analysis--')
    parser.add_argument('--config',type=str,default='configs/ult_crash_yolov8.yaml')
    args = parser.parse_args()
    cfg = yaml_load(args.config)
    cfg = Config(cfg)
    out_root = os.path.join('runs',cfg.task_name,'analysis')
    os.makedirs(out_root,exist_ok=True)
    #todo: setup logger
    setup_logger(out_root, distributed_rank=0, filename=f'{cfg.task_name}.txt', mode='a')
    ana_cfg = cfg.detect.anaysis
    model = YOLO(ana_cfg.ckpt)
    if cfg.benchmark_setting.name=='mot17':
        name = 'mot'
    else:
        name = copy.deepcopy(cfg.benchmark_setting.name)
    img_root = os.path.join('datasets',name,'train' if not cfg.benchmark_setting.test else 'test' )
    gt_path = os.path.join('datasets',ana_cfg.data_root,'annotations',ana_cfg.json_file)
    data = json.load(open(gt_path,'r'))
    file_list = [os.path.join(img_root,x['file_name']) for x in data['images']]
    id_list = [x['id'] for x in data['images']]
    logger.info(f'Images: {len(file_list)}')
    logger.info(f'{emoji.emojize(":rocket:")*3}=>model has been built!')
    #获取评估的json文件
    if ana_cfg.json and not os.path.exists(os.path.join(out_root,'res.json')):
        logger.info(f'{emoji.emojize(":rocket:")*3}=>Get JSON')
        _results = []
        batch_id = np.arange(len(file_list))//ana_cfg.batch
        nb = max(batch_id)+1
        file_list = [list(np.array(file_list)[batch_id==i]) for i in range(nb)]
        id_list = [list(np.array(id_list)[batch_id==i]) for i in range(nb)]
        progress_bar = tqdm
        for idx in progress_bar(range(len(file_list))):
            ids = id_list[idx]
            imgs = file_list[idx]
            results = model.predict(source=imgs,device='cuda:0',iou=ana_cfg.iou,
                                    conf=ana_cfg.conf,max_det=300,verbose=False)
            for id,res in zip(ids,results):
                boxes = res.boxes.xyxy.detach().to('cpu').numpy()
                clss = res.boxes.cls.detach().to('cpu').numpy()
                conf = res.boxes.conf.detach().to('cpu').numpy()
                boxes[:,2:4] -= boxes[:,:2]
                for b, c, s in zip(boxes, clss, conf):
                    b = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                    _results.append({"image_id": int(id), 'category_id': int(c)+1,
                                     'bbox': b, 'score': float(s)})
        json.dump(_results,open(os.path.join(out_root,'res.json'),'w'))
    logger.info(f'{emoji.emojize(":clown_face:")*3}=>Results JSON at {os.path.join(out_root,"res.json")}')

    #获取PR曲线-->todo: 评估差异是因为json文件未将坐标限制在图像内
    if ana_cfg.pr and os.path.isfile(os.path.join(out_root,'res.json')) and not os.path.exists(os.path.join(out_root,f'PR@{ana_cfg.thr}.xlsx')):
        logger.info(f'{emoji.emojize(":rocket:")*3}=>Get PR_CURVE')
        coco = COCO(gt_path)
        coco_dt = coco.loadRes(os.path.join(out_root,'res.json'))
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        p_value = coco_eval.eval['precision']
        recall = np.mat(np.arange(0.0, 1.01, 0.01)).T
        max_dets = -1
        if ana_cfg.thr==-1:
            map_all_pr = np.mean(p_value[:, :, :, 0, max_dets], axis=0)
        else:
            T = int((ana_cfg.thr - 0.5) / 0.05)
            map_all_pr = p_value[T, :, :, 0, max_dets]
        data = np.hstack((np.hstack((recall, map_all_pr)),
                          np.mat(np.mean(map_all_pr, axis=1)).T))
        df = pd.DataFrame(data)
        save_path = os.path.join(out_root,f'PR@{ana_cfg.thr}.xlsx')
        df.to_excel(save_path, index=False)
        logger.info(f'{emoji.emojize(":clown_face:")*3}=>PR CURVES at {save_path}')

    #获取热图
    demo_out = os.path.join(out_root,'images')
    os.makedirs(demo_out,exist_ok=True)
    if ana_cfg.heat and len(os.listdir(demo_out))==0:
        heater = Yolov8_heatmap(weight=ana_cfg.ckpt,device='cuda:0',method=ana_cfg.method,
                                layer=ana_cfg.layer,backward_type=ana_cfg.backward_type,conf_threshold=ana_cfg.vis_conf,
                                ratio=ana_cfg.ratio,show_box=ana_cfg.show_box,renormalize=ana_cfg.renormalize)
        logger.info(f'{emoji.emojize(":rocket:")*3}=>GRAD CAM is built')
        heater(ana_cfg.demo_dir,demo_out,ana_cfg.imgsz)
        logger.info(f'{emoji.emojize(":clown_face:")}=>{len(os.listdir(ana_cfg.demo_dir))} images have been saved at {demo_out}')

if __name__ == '__main__':
    main()