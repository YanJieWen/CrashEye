'''
@File: convert_crash_to_coco.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 9月 03, 2025
@HomePage: https://github.com/YanJieWen
'''
import os
import numpy as np
import json
import cv2

DATA_PATH = './datasets/Crash-Seq'
OUT_PATH = os.path.join(DATA_PATH,'annotations')
SPLITS = ['train_half','val_half','train','test']
HALF_VIDEO = True
CREATED_SPLITTED_ANN = True


if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH, exist_ok=True)
    for split in SPLITS:
        if split == 'test':
            data_path = os.path.join(DATA_PATH, 'test')
        else:
            data_path = os.path.join(DATA_PATH, 'train')
        out_path = os.path.join(OUT_PATH, f'{split}.json')
        out = {'images': [], 'annotations': [], 'videos': [], 'categories': [{'id': 1, 'name': 'point'}]}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_cur = 0
        tid_last = -1
        for seq in seqs:
            video_cnt += 1
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            images = os.listdir(img_path)
            num_images = len([x for x in images if 'jpg' in x])
            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]
            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                img = cv2.imread(os.path.join(data_path, f'{seq}/img1/{str(i + 1).zfill(6)}.jpg'))
                height, width = img.shape[:2]
                image_info = {'file_name': f'{seq}/img1/{str(i + 1).zfill(6)}.jpg', 'id': image_cnt + i + 1,
                              'frame_id': i + 1 - image_range[0], 'prev_image_id': image_cnt + i if i > 0 else -1,
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
            print(f'{seq}:{num_images} images')
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            if CREATED_SPLITTED_ANN and ('half' in split):
                anns_out = np.array([anns[i] for i in range(anns.shape[0]) if
                                     int(anns[i][0]) - 1 >= image_range[0] and int(anns[i][0]) - 1 <= image_range[1]],
                                    np.float32)
                anns_out[:, 0] -= image_range[0]
                gt_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(split))
                fout = open(gt_out, 'w')
                for o in anns_out:
                    fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                        int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                        int(o[6]), int(o[7]), o[8]))
                fout.close()
            print(f'{int(anns[:, 0].max())} ann images')
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                    continue
                track_id = int(anns[i][1])
                cat_id = int(anns[i][7])
                ann_cnt += 1
                category_id = 1
                ann = {
                    'id': ann_cnt,
                    'category_id': category_id,
                    'image_id': image_cnt + frame_id,
                    'track_id': track_id,
                    'bbox': anns[i][2:6].tolist(),
                    'conf': float(anns[i][6]),
                    'iscrowd': 0,
                    'area': float(anns[i][4] * anns[i][5])
                }
                out['annotations'].append(ann)
            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))