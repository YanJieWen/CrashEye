'''
@File: convert_coco_to_yolo.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 30, 2025
@HomePage: https://github.com/YanJieWen
'''

import os
import json
from tqdm import tqdm
import numpy as np
import sys
import shutil

DATATYPE = 'Crash-Seq'
SPLITS = ['train','val_half','test']
IMAGE_ROOT = 'datasets/mix_crash_ch/'
DATA_PATH = 'datasets/mix_crash_ch/annotations'
OUT_PATH = 'datasets/mix_ablation_crash_yolo/'


def convert(size, box):
    box = np.array(box)
    w,h = size
    box[2:] += box[:2]
    box[0] = np.clip(box[0], 0, w - 1)
    box[1] = np.clip(box[1], 0, h - 1)
    box[2] = np.clip(box[2], 1, w)
    box[3] = np.clip(box[3], 1, h)
    box[2:] -= box[:2]

    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    xywh = np.array([x,y,w,h])
    xywh = np.clip(xywh,0,1)
    x,y,w,h = xywh
    return (x, y, w, h)
if __name__ == '__main__':
    [os.makedirs(os.path.join(OUT_PATH, f'images/{split}'), exist_ok=True) for split in SPLITS]
    [os.makedirs(os.path.join(OUT_PATH, f'labels/{split}'), exist_ok=True) for split in SPLITS]
    for split in SPLITS:
        anns = json.load(open(os.path.join(DATA_PATH,f'{split}.json'),'r'))
        id_map = {}
        for i, cat in enumerate(anns['categories']):
            id_map[cat['id']] = i
        img_dict = {x['id']: i for i, x in enumerate(anns['images'])}
        img_ann_dict = {k: [] for k in img_dict.keys()}
        for i, ann in enumerate(anns['annotations']):
            img_ann_dict[ann['image_id']].append(i)
        pbar = tqdm(anns['images'], desc='caching image-annotations...', file=sys.stdout)
        for img in pbar:
            filename = img['file_name']
            img_width = img["width"]
            img_height = img["height"]
            img_id = img["id"]
            ann_txt_name = str(img_id).zfill(8) + '.txt'
            # 将图片复制到images中
            if split=='train':
                old_img_name = os.path.join(IMAGE_ROOT, filename)
            elif split=='val_half':
                old_img_name = os.path.join('datasets',DATATYPE,'train',filename)
            elif split=='test':
                old_img_name = os.path.join('datasets',DATATYPE,'test',filename)
            else:
                pass
            new_img_name = os.path.join(OUT_PATH, f'images/{split}/{str(img_id).zfill(8)}.jpg')
            shutil.copy(old_img_name, new_img_name)
            f_txt = open(os.path.join(OUT_PATH, f'labels/{split}/{ann_txt_name}'), 'w')
            for ann_id in img_ann_dict[img_id]:
                ann = anns['annotations'][ann_id]
                box = convert((img_width, img_height), ann['bbox'])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]],
                                                  box[0], box[1], box[2], box[3]))
            f_txt.close()