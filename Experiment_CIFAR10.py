

import numpy as np

import keras.activations
from keras import backend as K
from keras.engine.topology import Layer

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import *

from TTLayer import *

tt_input_shape=[8, 8, 8, 6]
tt_output_shape=[10, 10, 10, 10]
tt_ranks=[1, 2, 2, 2, 1]

input = Input((32, 32, 3))
ttl = TT_Layer(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
               tt_ranks=tt_ranks,
               activation='relu', use_bias=True,
               kernel_regularizer=regularizers.l2(.001), )
z1 = ttl(input)
z1 = Dropout(.5)(z1)
output = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(.01),)(z1)

model = Model(input, output)

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=1000, validation_data=[x_test, y_test], verbose=2)

