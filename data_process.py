from bs4 import BeautifulSoup
import os
import cv2


def process():
	# train_result用于后续yolo训练,k_mean_result用于计算anchor boxes
	train_result = []
	k_mean_result = []

	# 遍历图片
	for i in all_name:
		image_name = image_path+i.rstrip('xml')+'jpg'
		train_result.append(image_name)
		soup = BeautifulSoup(open(file_path+i), 'lxml')
		bbx = soup.find_all('object')
		img = cv2.imread(image_name)
		height, width, _ = img.shape

		# 遍历候选框
		for j in bbx:
			name = str(j.contents[1].string)
			if name != 'waterweeds':
				xmin = int(j.xmin.string)
				ymin = int(j.ymin.string)
				xmax = int(j.xmax.string)
				ymax = int(j.ymax.string)
				index = str(class_label.index(name))

				# 防止大小混淆
				if xmin > xmax:
					xmax, xmin = xmin, xmax
				if ymin > ymax:
					ymax, ymin = ymax, ymin

				# 判断ground truth是否超出图像范围
				if xmin > width or xmax <= 0:
					continue
				if ymin > height or ymax <= 0:
					continue

				# 处于图像边缘候选框的处理
				if xmin <= 0:
					xmin = 1
				if ymin <= 0:
					ymin = 1
				if xmax > width:
					xmax = width
				if ymax > height:
					ymax = height

				# 判断是否为无效框
				x_distance = xmax - xmin
				y_distance = ymax - ymin
				if x_distance == 0 or y_distance == 0:
					continue

				xmin, xmax, ymin, ymax = str(xmin), str(xmax), str(ymin), str(ymax)
				train_result.append(' ')
				for x in [xmin, ymin, xmax, ymax]:
					train_result.append(x)
					k_mean_result.append(x)
					train_result.append(',')
					if x is not ymax:
						k_mean_result.append(',')
					else:
						train_result.append(index)
				k_mean_result.append('\n')
		train_result.append('\n')

	return train_result, k_mean_result


def save_file(train_result, k_mean_result):
	with open('data/train_data.txt', 'w') as f:
		f.writelines(train_result)
	with open('data/k_mean_data.txt', 'w') as f:
		f.writelines(k_mean_result)


if __name__ == '__main__':
	class_label = ['holothurian', 'echinus', 'scallop', 'starfish']
	file_path = 'data/train/box/'
	image_path = 'data/train/image/'
	all_name = os.listdir(file_path)

	train_result, k_mean_result = process()
	save_file(train_result, k_mean_result)
