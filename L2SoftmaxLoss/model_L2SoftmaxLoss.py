# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import backend as K


# # pretrain with L2 Softmax Loss

def make_model(input_shape, num_classes, alpha=0.3):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    # l2 softmax
    model.add(Dense(128, activation='linear', 
                                    activity_regularizer=regularizers.l2(alpha)
                   ))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# # Post-Pretrain

def cosine_distance(inputs):
    x1, x2 = inputs
    x1 = K.l2_normalize(x1, axis=-1)
    x2 = K.l2_normalize(x2, axis=-1)
    return K.sum(x1 * x2, axis=-1, keepdims=True)


def make_metric_model(base_model):
    layer_name = base_model.layers[-2].name
    base_model.layers[-2].activity_regularizer = None

    x1_input = Input(shape=base_model.input.shape[1:].as_list())
    x1 = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)(x1_input)

    x2_input = Input(shape=base_model.input.shape[1:].as_list())
    x2 = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)(x2_input)

    distance = Lambda(cosine_distance)([x1, x2])
    model = Model(inputs=[x1_input, x2_input], outputs=distance)

    return model
