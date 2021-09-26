import numpy as np
import cv2
import os

class_dictionary = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                    10: 'cow', 11: 'dining_table', 12: 'dog', 13: 'horse', 14: 'motorbike',
                    15: 'person', 16: 'potted_plant', 17: 'sheep', 18: 'sofa', 19: 'train',
                    20: 'TV_monitor', 21: 'edge'}


# VOC2007-2012数据集简介：

# 两个文件夹：JPEGImages文件夹包含17125张rgb图片，SegmentationClass文件夹包含12031张语义分割图片，id序号相等对应标注同一张。
# 利用from PIL import Image函数读取SegmentationClass中png图片，可以得到标注信息。
# Image读取得到 (h，w) 单通道矩阵，像素值总共有22个类别，由22个数字代替：0、1、2、...、20、和255。

# 0代表背景信息
# 1-20代表图片中目标物体种类
# 255代表目标物体轮廓信息，在代码处理过程中我们将其忽略

def read_path():

    data_x = []
    data_y = []

    filename = os.listdir('SegmentationClass')
    filename.sort()
    for name in filename:

        serial_number = name.split('.')[0]
        img_path = 'JPEGImages/' + serial_number + '.jpg'
        seg_path = 'SegmentationClass/' + serial_number + '.png'

        data_x.append(img_path)
        data_y.append(seg_path)

    return data_x, data_y


def make_data():

    data_x, data_y = read_path()
    print('all image quantity : ', len(data_y))    # 12031

    train_x = data_x[:11000]
    train_y = data_y[:11000]
    val_x = data_x[11000:]
    val_y = data_y[11000:]
    test_x = data_x[11000:]
    test_y = data_y[11000:]

    return train_x, train_y, val_x, val_y, test_x, test_y

