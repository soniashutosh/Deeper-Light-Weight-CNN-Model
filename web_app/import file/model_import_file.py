import tensorflow
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import time
import numpy as np

def model(input_shape,output_shape):
    def bn_rl_conv(x, filters, kernel_size):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if kernel_size == 1:
            x = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same')(x)
        else:
            x = Conv2D(filters=filters,
                    kernel_size=(1,kernel_size),
                    padding='same')(x)
            x = Conv2D(filters=filters,
                    kernel_size=(kernel_size,1),
                    padding='same')(x)
        return x

    def collective_block(tensor, k, reps):
        feature_list = [tensor]
        for _ in range(reps):
            x = bn_rl_conv(tensor, filters=4*k, kernel_size=1)
            x = bn_rl_conv(x, filters=k, kernel_size=3)
#             tensor = Concatenate()([tensor, x])
            feature_list.append(x)
            tensor = Concatenate()(feature_list)
            
        return tensor

    def passage_layer(x, theta):
        f = int(tensorflow.keras.backend.int_shape(x)[-1] * theta)
        x = bn_rl_conv(x, filters=f, kernel_size=1)
        x = AvgPool2D(pool_size=2, strides=2, padding='same')(x)
        return x

    k = 32
    theta = 0.5
    repetitions = 6, 12, 24, 16
    
    input = Input(input_shape)
    
    x = Conv2D(2*k, 7, strides=2, padding='same')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    passage_layer_output = []
    for reps in repetitions:
        d = collective_block(x, k, reps)
        x = passage_layer(d, theta)
        length = len(passage_layer_output)
        if length > 0:
            ele = passage_layer_output[length-1]
            extr = passage_layer(ele, theta)
            x = Concatenate()([x, extr])
        passage_layer_output.append(x)
    
    x = GlobalAvgPool2D()(d)
    # Actual Activation function is softmax
    output = Dense(output_shape, activation='softmax')(x)
    model = Model(input, output)

    return model

def func():
    print("Heyy")