import sys
sys.path.insert(0,'data_augmention/')
from augment import augment
import os
import cv2
import numpy as np
from multiprocessing import Pool

number = 0
def img_aug(name, train=False, input_shape=(480,480)):
    global number

    img = cv2.imread(image_path+name)
    if train == False:
        h, w = input_shape
        ih, iw = img.shape[:2]
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        img = cv2.resize(img,(nw,nh))
        img = augment(img)

        # 用灰色像素块来做背景扩充图片满足输入尺寸需求
        dx = (w-nw) // 2
        dy = (h-nh) // 2
        img = np.pad(img, ((dy, dy), (dx, dx), (0, 0)),
                     'constant', constant_values=128)
        if tuple(img.shape[:2]) != input_shape:
            img = np.pad(img, ((0, input_shape[0]-img.shape[0]), 
                    (0, input_shape[1]-img.shape[1]), (0, 0)),
                    'constant', constant_values=128)
    else:
        img = augment(img)

    cv2.imwrite(result_path+name, img)
    number += 1
    print('{}. {} is finish!'.format(number, name))


if __name__ == '__main__':
    image_path = 'data/test/test-B-image/'
    result_path = 'data/test/test_B_augment/'
    all_name = os.listdir(image_path)
    p = Pool(10)
    for i in all_name:
        p.apply_async(img_aug, args=(i,))
    p.close()
    p.join()
