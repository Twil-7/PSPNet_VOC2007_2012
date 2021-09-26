# PSPNet_VOC2007_2012

# 环境配置

python == 3.8

tensorflow == 2.4.1

keras == 2.4.3

如果想切换到1.6.0的tensoflow版本，匹配python==3.6，需要在代码这个tf命令中进行修改：

tf.image.resize(x, (K.int_shape(y)[1], K.int_shape(y)[2]))改成tf.image.resize_images

# 运行：

直接运行mian.py即可。

1、JPEGImages文件夹：原始数据集，内含17125张rgb图片，20类目标 + 1类背景 + 1类轮廓。

2、SegmentationClass文件夹：原始数据集，内含12031张rgb图片，标记信息用数字0-21和255表示。

数据集下载：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

3、demo文件夹：test_data的效果演示图，可以看出训练出来的非常不错，所有像素点分类精度接近91%。


Epoch 1/20
687/687 [==============================] - 1985s 3s/step - loss: 0.1634 - accuracy: 0.9302 - val_loss: 0.2126 - val_accuracy: 0.9077
Epoch 2/20
687/687 [==============================] - 1915s 3s/step - loss: 0.1589 - accuracy: 0.9309 - val_loss: 0.2122 - val_accuracy: 0.9079
Epoch 3/20
687/687 [==============================] - 2026s 3s/step - loss: 0.1585 - accuracy: 0.9309 - val_loss: 0.2100 - val_accuracy: 0.9088
Epoch 4/20
687/687 [==============================] - 2132s 3s/step - loss: 0.1562 - accuracy: 0.9322 - val_loss: 0.2114 - val_accuracy: 0.9083


4、download_weights.h5：迁移学习，网上下载backbone的部分权重。

5、mobile_netv2.py：特征提取网络，PSPNet的backbone部分。

6、psp_model.py：PSPNet整体结构。

7、utils.py：重构的loss损失函数。

8、Logs文件夹：记录每大轮下每个epoch训练好的权重文件。
