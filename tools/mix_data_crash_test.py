'''
@File: mix_data_crash_test.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 9月 03, 2025
@HomePage: https://github.com/YanJieWen
'''

import json
import os
'''
Crash2024+Train-->test
cd datasets
mkdir -p mix_crash_det/annotations
cp Crash-seq/annotations/val_half.json mix_det/annotations/val_half.json
cp Crash-seq/annotations/test.json mix_det/annotations/test.json
cd mix_det
ln -s ../Crash-seq/train crash_train
ln -s ../crash2024/train crash2024_train
ln -s ../crash2024/val crash2024_val
ln -s ../crash2024/test crash2024_test
'''

crash_json = json.load(open('./datasets/Crash-Seq/annotations/train.json','r'))
img_list = list()
for img in crash_json['images']:
    img['file_name'] = 'crash_train/'+img['file_name']
    img_list.append(img)
ann_list = list()
for ann in crash_json['annotations']:
    ann_list.append(ann)
video_list = crash_json['videos']
category_list = crash_json['categories']

print(f'Crash-seq:{len(crash_json["images"])} images \t {len(crash_json["annotations"])} annotations')


max_img = 10000
max_ann = 300000
max_video = 10
crash_staic_json = json.load(open('./datasets/crash2024/annotations/train.json','r'))
img_id_count = 0
for img in crash_staic_json['images']:
    img_id_count += 1
    img['file_name'] = 'crash2024_train/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
for ann in crash_staic_json['annotations']:
    if ann['category_id']==1:
        ann['id'] = ann['id'] + max_ann
        ann['image_id'] = ann['image_id'] + max_img
        ann_list.append(ann)
video_list.append({
    'id': max_video,
    'file_name': 'crash2024_train'
})
print(f'crash2024_train: {len(crash_staic_json["images"])} images \t {len(crash_staic_json["annotations"])} annotations')


max_img = 100000
max_ann = 1000000
crash_staic_val_json = json.load(open('./datasets/Crash2024/annotations/val.json','r'))
img_id_count = 0
for img in crash_staic_val_json['images']:
    img_id_count += 1
    img['file_name'] = 'crash2024_val/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
for ann in crash_staic_val_json['annotations']:
    if ann['category_id']==1:
        ann['id'] = ann['id'] + max_ann
        ann['image_id'] = ann['image_id'] + max_img
        ann_list.append(ann)
video_list.append({
    'id': max_video,
    'file_name': 'crash2024_val'
})
print(f'crash2024_val: {len(crash_staic_val_json["images"])} images \t {len(crash_staic_val_json["annotations"])} annotations')

max_img = 200000
max_ann = 2000000
crash_static_test_json = json.load(open('./datasets/Crash2024/annotations/test.json','r'))
img_id_count = 0
for img in crash_static_test_json['images']:
    img_id_count += 1
    img['file_name'] = 'crash2024_test/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
for ann in crash_static_test_json['annotations']:
    if ann['category_id']==1:
        ann['id'] = ann['id'] + max_ann
        ann['image_id'] = ann['image_id'] + max_img
        ann_list.append(ann)
video_list.append({
    'id': max_video,
    'file_name': 'crash2024_test'
})
print(f'crash2024_test: {len(crash_static_test_json["images"])} images \t {len(crash_static_test_json["annotations"])} annotations')

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
print(f'{len(img_list)} images, {len(ann_list)} instances')
json.dump(mix_json,open('./datasets/mix_crash_det/annotations/train.json','w'))
