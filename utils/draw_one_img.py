'''
@File: draw_one_img.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 23, 2025
@HomePage: https://github.com/YanJieWen
'''

import numpy as np

import PIL.ImageDraw as Imagedraw
from PIL import ImageColor,ImageFont
from PIL import Image
from .colors import colors

def draw_text(draw,box,cls,score,cls_dict,color,font="msyh.ttc",font_size=18):
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()
    left, top, right, bottom = box
    display_str = f"{cls_dict[int(cls)]}: {int(100 * score)}%" if cls_dict is not None else \
        f'{int(cls)}: {int(100 * score)}%'
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)
    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                ds,
                fill='black',
                font=font)
        left += text_width

def draw(res,img,cls_dict,draw_thresh=0.3):
    _colors = [ImageColor.getrgb(colors[cls%len(colors)]) for cls in np.asarray(res[:,-1],dtype=int)]
    draw = Imagedraw.Draw(img)
    res = res[res[:,-2]>draw_thresh,:]
    for re,color in zip(res,_colors):
        x1,y1,x2,y2,score,cls = re
        draw.line([(x1, y1), (x1, y2), (x2, y2),
                        (x2, y1), (x1, y1)], width=5, fill=color)
        draw_text(draw,[x1,y1,x2,y2],cls,score,cls_dict,color)


def draw_traj(img,tracklets,traj_length=30):
    draw = Imagedraw.Draw(img)
    for id,trajs in tracklets.items():
        if len(trajs)>1:
            for j in range(1,len(trajs)):
                draw.line([(int(trajs[j-1][0]),int(trajs[j-1][1])),
                           (int(trajs[j][0]),int(trajs[j][1]))],width=3,fill=ImageColor.getrgb(colors[int(id) % len(colors)]))
        if len(trajs)>traj_length:
            trajs.pop(0)