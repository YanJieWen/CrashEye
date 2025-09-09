'''
@File: mot.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 26, 2025
@HomePage: https://github.com/YanJieWen
'''

from torch.utils.data import Dataset

import cv2
import numpy as np
from pycocotools.coco import COCO

import os

class MOTDataset(Dataset):
    def __init__(self,data_dir=None,json_file='train_half.json',name='train',img_size=(608,1088),preproc=None):
        '''

        Args:
            data_dir: benchmark的根目录
            json_file: 注释文件名称
            name: 图像存储根目录
            img_size: 需要缩放的图片大小
            preproc: 处理pipleline-->将tgt置为空置
        '''
        super().__init__()
        if data_dir is None:
            data_dir = os.path.join('datasets','mot')
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir,'annotations',self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])

        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_idx) for _idx in self.ids]

    def load_anno_from_ids(self,id):
        im_ann =self.coco.loadImgs(id)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
        num_objs = len(objs)
        res = np.zeros((num_objs, 6))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]
        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)
        del im_ann, annotations
        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )
        img = cv2.imread(img_file)
        assert img is not None

        return img, res.copy(), img_info, np.array([id_])

    def __getitem__(self, idx):
        img, target, img_info, img_id = self.pull_item(idx)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.img_size)
        return img, target, img_info, img_id #chw-->rgb/xyxyc_intt_int


# if __name__ == '__main__':
#
#     from data import ValTransform
#     from utils import draw
#     from PIL import Image
#     import matplotlib.pyplot as plt
#     preproc = ValTransform(
#         rgb_means=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#     )
#     data = MOTDataset(data_dir=None,json_file='train_half.json',name='train',img_size=(608,1088),preproc=None)
#     img,tgt,_,_ = data[0]
#     cls_dict = {v:k for k,v in enumerate(tgt[:,-1])}
#     tgt[:,-2] = 1
#     img = Image.fromarray(img[...,::-1])
#     draw(tgt, img, cls_dict=cls_dict)
#     plt.imshow(img)
#     plt.show()

