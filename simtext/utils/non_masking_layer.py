# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: non_masking_layer.py
# time: 2019-05-23 14:05
import os

os.environ['TF_KERAS'] = '1'
import keras_bert

custom_objects = keras_bert.get_custom_objects()
from tensorflow.python.keras.layers import Layer


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x


custom_objects['NonMaskingLayer'] = NonMaskingLayer


if __name__ == "__main__":
    print("Hello world")
