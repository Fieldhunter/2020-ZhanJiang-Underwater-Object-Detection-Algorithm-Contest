"""Miscellaneous utility functions."""

from functools import reduce
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import albumentations as A
import random

import sys
sys.path.insert(0,'../data_augmention/')
from augment import augment

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
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = cv2.imread(line[0])
    ih, iw = image.shape[:2]
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    # resize image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh))
    
    image = augment(image)
    dx = (w-nw)//2
    dy = (h-nh)//2
    box[:, [0,2]] = box[:, [0,2]]*scale + dx
    box[:, [1,3]] = box[:, [1,3]]*scale + dy


    # # flip image or not
    # flip = rand()<.5
    # if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # # distort image
    # hue = rand(-hue, hue)
    # sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    # val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    # x = rgb_to_hsv(np.array(image)/255.)
    # x[..., 0] += hue
    # x[..., 0][x[..., 0]>1] -= 1
    # x[..., 0][x[..., 0]<0] += 1
    # x[..., 1] *= sat
    # x[..., 2] *= val
    # x[x>1] = 1
    # x[x<0] = 0
    # image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    
    image = np.pad(image, ((dy, dy), (dx, dx), (0, 0)), 'constant', constant_values=128)
    if tuple(image.shape[:2]) != (416,416):
        image = np.pad(image, ((0, 416-image.shape[0]), (0, 416-image.shape[1]), (0, 0)),\
            'constant', constant_values=128)

    # correct boxes
    # box_data = np.zeros((max_boxes,5))
    # if len(box)>0:
    #     np.random.shuffle(box)
    #     if flip: box[:, [0,2]] = w - box[:, [2,0]]
    #     box[:, 0:2][box[:, 0:2]<0] = 0
    #     box[:, 2][box[:, 2]>w] = w
    #     box[:, 3][box[:, 3]>h] = h
    #     box_w = box[:, 2] - box[:, 0]
    #     box_h = box[:, 3] - box[:, 1]
    #     box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
    #     if len(box)>max_boxes: box = box[:max_boxes]
    #     box_data[:len(box)] = box

    return image_data, box
