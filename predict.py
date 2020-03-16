import colorsys
import os
import glob
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import cv2
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
import pandas as pd


class YOLO(object):
    _defaults = {
        "model_path": 'models/yolo/trained_weights_final.h5',
        "anchors_path": 'data/yolo_anchors.txt',
        "classes_path": 'data/classes.txt',
        "score" : 0.05,
        "iou" : 0.45,
        "model_image_size" : (448, 448),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self):
        self.__dict__.update(self._defaults) # set up default values
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = np.array(image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[1], image.shape[0]],
                K.learning_phase(): 0
            })

        return out_classes, out_scores, out_boxes


def detect_img(yolo, test):
    class_label = ['holothurian', 'echinus', 'scallop', 'starfish']
    name, image_id, confidence, xmin, ymin, xmax, ymax = \
        [], [], [], [], [], [], []

    for img in test:
        image = cv2.imread(img)
        height, width, _ = cv2.imread(img.replace('test_augment', 'test-A-image')).shape
        scale = min(448/width, 448/height)
        nw = int(width*scale)
        nh = int(height*scale)
        dx = (448-nw) // 2
        dy = (448-nh) // 2
        out_classes, out_scores, out_boxes = yolo.detect_image(image)

        for v, i in enumerate(out_boxes):
            # 水草一类删除
            if out_classes[v] == 4:
                continue

            ym, xm, ya, xa = i
            # 范围修正
            xm, ym, xa, ya = (xm-dx)/scale+1, (ym-dy)/scale+1, (xa-dx)/scale+1, (ya-dy)/scale+1

            # 防止大小混淆
            if xm > xa:
                xa, xm = xm, xa
            if ym > ya:
                ya, ym = ym, ya

            # 判断ground truth是否超出图像范围
            if xm > width or xa < 1:
                continue
            if ym > height or ya < 1:
                continue

            # 处于图像边缘候选框的处理
            if xm < 1:
                xm = 1
            if ym < 1:
                ym = 1
            if xa > width:
                xa = width
            if ya > height:
                ya = height

            # 判断是否为无效框
            x_distance = xa - xm
            y_distance = ya - ym
            if x_distance == 0 or y_distance == 0:
                continue

            # 四舍五入转换为int
            xm = int(round(xm))
            ym = int(round(ym))
            xa = int(round(xa))
            ya = int(round(ya))

            xmin.append(xm)
            ymin.append(ym)
            xmax.append(xa)
            ymax.append(ya)

            name.append(class_label[out_classes[v]])
            confidence.append(out_scores[v])
            image_id.append(img.replace('jpg', 'xml').lstrip(TEST_PATH))

    save_csv(name, image_id, confidence, xmin, ymin, xmax, ymax)


def save_csv(name, image_id, confidence, xmin, ymin, xmax, ymax):
    result_table = pd.DataFrame({"name": name,
                                 "image_id": image_id, 
                                 "confidence":confidence, 
                                 "xmin":xmin, 
                                 "ymin":ymin, 
                                 "xmax":xmax, 
                                 "ymax":ymax})
    result_table.to_csv("predict.csv", index=False)


if __name__ == '__main__':
    TEST_PATH = "data/test_augment/"
    TEST_NAME = glob.glob(TEST_PATH + "*.jpg")
    yolo = YOLO()

    detect_img(yolo, TEST_NAME)