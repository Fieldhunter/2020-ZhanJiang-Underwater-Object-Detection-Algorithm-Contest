from bs4 import BeautifulSoup
import os
import cv2
from tqdm import tqdm


def process():
    # train_result用于后续yolo训练,k_mean_result用于计算anchor boxes
    train_result = []
    k_mean_result = []

    # 遍历图片
    for i in tqdm(all_name):
        # 同时对范围进行修正，像素起点改为0
        image_name = image_path+i.rstrip('xml')+'jpg'
        img_name_append = False
        soup = BeautifulSoup(open(file_path+i), 'lxml')
        bbx = soup.find_all('object')
        img = cv2.imread(image_name)
        height, width, _ = img.shape

        # 遍历候选框
        for j in bbx:
            # 同时对范围进行修正，像素起点改为0
            name = str(j.contents[1].string)
            xmin = int(j.xmin.string) - 1
            ymin = int(j.ymin.string) - 1
            xmax = int(j.xmax.string) - 1
            ymax = int(j.ymax.string) - 1
            index = str(class_label.index(name))

            # 防止大小混淆
            if xmin > xmax:
                xmax, xmin = xmin, xmax
            if ymin > ymax:
                ymax, ymin = ymin, ymax

            # 判断ground truth是否超出图像范围
            if xmin > width - 1 or xmax < 0:
                continue
            if ymin > height - 1 or ymax < 0:
                continue

            # 处于图像边缘候选框的处理
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > width - 1:
                xmax = width - 1
            if ymax > height - 1:
                ymax = height - 1

            # 判断是否为无效框
            x_distance = xmax - xmin
            y_distance = ymax - ymin
            if x_distance == 0 or y_distance == 0:
                continue

            # 对于过小的候选框进行删除
            square = (xmax - xmin) * (ymax - ymin)
            if square < 120:
                continue

            # 防止有图片没有任何候选框
            if img_name_append == False:
                train_result.append(image_name)
                img_name_append = True

             # 判断是否增加kmean缩放候选框
            scale = min(448/width, 448/height)
            square = ((xmax - xmin)*scale) * ((ymax - ymin)*scale)
            if square < 120:
                kmean_scale = False
            else:
                kmean_scale = True

            xmin, xmax, ymin, ymax = str(xmin), str(xmax), str(ymin), str(ymax)
            train_result.append(' ')
            for x in [xmin, ymin, xmax, ymax]:
                train_result.append(x)
                train_result.append(',')
                if x is ymax:
                    train_result.append(index)

            # kmean添加缩放候选框
            if kmean_scale == True:
                for x in [xmin, ymin, xmax, ymax]:
                    k_mean_result.append(str(int(round(int(x)*scale))))
                    if x is not ymax:
                        k_mean_result.append(',')
                    else:
                        k_mean_result.append('\n')

        if img_name_append == True:
            train_result.append('\n')

    return train_result, k_mean_result


def save_file(train_result, k_mean_result):
    with open('../data/train_data.txt', 'w') as f:
        f.writelines(train_result)
    with open('../data/k_mean_data.txt', 'w') as f:
        f.writelines(k_mean_result)


if __name__ == '__main__':
    class_label = ['holothurian', 'echinus', 'scallop', 'starfish', 'waterweeds']
    file_path = '../data/train/box/'
    image_path = '../data/train/augment/'
    all_name = os.listdir(file_path)

    train_result, k_mean_result = process()
    save_file(train_result, k_mean_result)
