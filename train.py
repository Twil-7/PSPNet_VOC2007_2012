import cv2
import os
import random
import numpy as np
from keras.utils import Sequence
import math
from psp_model import get_psp_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utils import my_cross_entropy
from PIL import Image


class_dictionary = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                    10: 'cow', 11: 'dining_table', 12: 'dog', 13: 'horse', 14: 'motorbike',
                    15: 'person', 16: 'potted_plant', 17: 'sheep', 18: 'sofa', 19: 'train',
                    20: 'TV_monitor', 21: 'edge'}


inputs_size = (473, 473, 3)
n_classes = 20 + 1


class SequenceData(Sequence):

    def __init__(self, data_x, data_y, batch_size):
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.indexes = np.arange(len(self.data_x))

    def __len__(self):
        return math.floor(len(self.data_x) / float(self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):

        batch_index = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.data_x[k] for k in batch_index]
        batch_y = [self.data_y[k] for k in batch_index]

        x = np.zeros((self.batch_size, inputs_size[1], inputs_size[0], 3))
        y = np.zeros((self.batch_size, inputs_size[1], inputs_size[0], n_classes + 1))

        for i in range(self.batch_size):

            img = cv2.imread(batch_x[i])
            img1 = cv2.resize(img, (inputs_size[1], inputs_size[0]), interpolation=cv2.INTER_AREA)
            img2 = img1 / 255
            x[i, :, :, :] = img2

            seg = Image.open(batch_y[i])
            seg = np.array(seg)
            seg1 = cv2.resize(seg, (473, 473), interpolation=cv2.INTER_NEAREST)

            # 将对应edge像素区域的255数值信息，全部转化为数值21，然后利用eye函数转换成语义分割对应的数据结构形式
            seg1[seg1 == 255] = 21

            # 20类目标物体 + 1背景类别 + 1轮廓类别
            seg2 = np.eye(n_classes + 1)[seg1.reshape([-1])]                        # (223729, 22)
            seg3 = seg2.reshape((inputs_size[1], inputs_size[0], n_classes + 1))    # (473, 473, 22)
            y[i, :, :, :] = seg3

            # 用来测试读取的label是否会出错的，demon记录该图像上所有类别的种类

            # demon = []
            # for i1 in range(seg1.shape[0]):
            #     for j1 in range(seg1.shape[1]):
            #         demon.append(seg1[i1, j1])
            # print(set(demon))

            # cv2.namedWindow("Image")
            # cv2.imshow("Image", img2)
            # cv2.waitKey(0)

            # cv2.namedWindow("seg1")
            # cv2.imshow("seg1", seg1/21)
            # cv2.waitKey(0)

        return x, y


def train_network(train_generator, validation_generator, epoch):

    model = get_psp_model()
    model.load_weights('download_weights.h5', by_name=True, skip_mismatch=True)
    print('PSPNet网络层总数为：', len(model.layers))    # 175

    freeze_layers = 146
    for i in range(freeze_layers):
        model.layers[i].trainable = False
        print(model.layers[i].name)

    adam = Adam(lr=1e-4)
    log_dir = "Logs/1/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}_val_accuracy{val_accuracy:.5f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    # 这里特意重构了一遍交叉熵损失函数，把edge边缘信息那一类的损失计算给忽略掉了
    model.compile(loss=my_cross_entropy(), optimizer=adam, metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights('first_weights.hdf5')


def load_network_then_train(train_generator, validation_generator, epoch, input_name, output_name):

    model = get_psp_model()
    model.load_weights(input_name)
    print('PSPNet网络层总数为：', len(model.layers))  # 175

    freeze_layers = 146
    for i in range(freeze_layers):
        model.layers[i].trainable = False
        print(model.layers[i].name)

    adam = Adam(lr=1e-5)
    log_dir = "Logs/2/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}_val_accuracy{val_accuracy:.5f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    # 这里特意重构了一遍交叉熵损失函数，把edge边缘信息那一类的损失计算给忽略掉了
    model.compile(loss=my_cross_entropy(), optimizer=adam, metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights(output_name)
