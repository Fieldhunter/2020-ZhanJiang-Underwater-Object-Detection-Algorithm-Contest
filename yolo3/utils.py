"""Miscellaneous utility functions."""

from functools import reduce
from PIL import Image
import numpy as np
import cv2
import random
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomCrop,
    Resize,
    Compose
)


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[:2]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh))

    # 用灰色像素块来做背景扩充图片满足输入尺寸需求
    dx = (w-nw) // 2
    dy = (h-nh) // 2
    image = np.pad(image, ((dy, dy), (dx, dx), (0, 0)),
                   'constant', constant_values=128)
    if tuple(image.shape[:2]) != (448, 448):
        image = np.pad(image, ((0, input_shape[0]-image.shape[0]), 
                (0, input_shape[1]-image.shape[1]), (0, 0)),
                'constant', constant_values=128)
    return image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params={'format': 'pascal_voc', 
                                     'min_area': min_area, 
                                     'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})

def get_random_data(annotation_line, input_shape, max_boxes=100,trainable=True):
    # 读入图片，候选框和尺寸信息
    line = annotation_line.split()
    image = cv2.imread(line[0])
    ih, iw = image.shape[:2]
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    # 候选框和标签分离
    bboxes = box.tolist()
    category_id = []
    for n in range(len(bboxes)):
        category_id.append(bboxes[n][-1])
        del bboxes[n][-1]

    # 直接缩放或者随机裁剪来缩放图片，目的是保持图片原有的长宽比
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    if trainable:
        if rand() < .5:
            annotations = {'image': image,
                           'bboxes':bboxes,
                           'category_id': category_id}
            aug = get_aug([Resize(p=1, height=nh, width=nw)])
            augmented = aug(**annotations)

            # 删除面积过小的候选框
            bboxes = augmented['bboxes']
            category_id = augmented['category_id']
            for n in range(len(bboxes)):
                if n >= len(bboxes):
                    break

                i = bboxes[n]
                if (i[2]-i[0]) * (i[3]-i[1]) < 120:
                    del bboxes[n]
                    del category_id[n]
            if len(bboxes) == 0:
                return None, None

            augmented['bboxes'] = bboxes
            augmented['category_id'] = category_id
        else:
            annotations = {'image': image,
                           'bboxes':bboxes,
                           'category_id': category_id}
            aug = get_aug([RandomCrop(p=1, height=nh, width=nw)], 
                           min_visibility=0.3)
            augmented = aug(**annotations)
            if len(augmented['bboxes']) == 0:
                return None, None

        # 水平和垂直翻转
        if rand() < .5:
            aug = get_aug([VerticalFlip(p=1)])
            augmented = aug(**augmented)
        if rand() < .5:
            aug = get_aug([HorizontalFlip(p=1)])
            augmented = aug(**augmented)
    else:
        annotations = {'image': image,
                       'bboxes':bboxes,
                       'category_id': category_id}
        aug = get_aug([Resize(p=1, height=nh, width=nw)])
        augmented = aug(**annotations)

    # 标签数组和候选框数组合并
    image = augmented['image']
    box = augmented['bboxes']
    category_id = augmented['category_id']
    for n in range(len(box)):
        box[n] = list(map(lambda x : round(x), box[n]))
        box[n].append(category_id[n])
    
    # 用灰色像素块来做背景扩充图片满足输入尺寸需求
    dx = (w-nw) // 2
    dy = (h-nh) // 2
    image = np.pad(image, ((dy, dy), (dx, dx), (0, 0)),
                   'constant', constant_values=128)
    if tuple(image.shape[:2]) != input_shape:
        image = np.pad(image, ((0, input_shape[0]-image.shape[0]), 
                (0, input_shape[1]-image.shape[1]), (0, 0)),
                'constant', constant_values=128)

    image = image / 255.
    box_data = np.zeros((max_boxes,5))
    box = np.array([np.array(box[i]) for i in range(len(box))])
    box[:, [0,2]] = box[:, [0,2]] + dx
    box[:, [1,3]] = box[:, [1,3]] + dy
    box_data[:len(box)] = box

    return image, box_data
