import tensorflow as tf
from keras import backend as k


def my_cross_entropy():

    def _cross_entropy(y_true, y_pre):

        # y_true  :  shape = (None, 473, 473, 22)
        # y_pre   :  shape = (None, 473, 473, 21)

        y_pre = k.clip(y_pre, k.epsilon(), 1.0 - k.epsilon())

        # 特意忽略掉edge边缘信息那一类的损失计算
        ce_loss = - y_true[..., :-1] * k.log(y_pre)
        ce_loss1 = k.mean(k.sum(ce_loss, axis=-1))

        return ce_loss1

    return _cross_entropy


