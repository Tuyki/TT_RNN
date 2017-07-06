__author__ = "Yinchong Yang"
__copyright__ = "Siemens AG, 2017"
__licencse__ = "MIT"
__version__ = "0.1"

"""
MIT License

Copyright (c) 2017 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""



"""
On the MNIST data we experiment two 2-layered NN models: The first model contains two dense layers,
while the second model replaces the first dense layer with TT layer.
It can be shown that the when applying a TT layer with significantly less parameters, one can speed
up the training and inference to a very large extent. In detail:

The first standard model has 1048576 parameters in the first layer. It takes ca 48 seconds to train
for one epoch. The accuracy after 50 epochs is 0.9686.
The second model with a TT layer contains 1248 weights and each epoch takes ca 9 seconds;
the accuracy after 50 epochs is 0.9785.
Compression factor = 1248 / 1048576 = 0.00119018554688

According to the original paper, the TT layer is considered to compress the otherwise dense layer.
In this case, however, due to the fact that the model with TT layer actually shows better performances,
"""

# Basic
import numpy as np

# Keras Model
from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import *

# TT Layer
from TTLayer import TT_Layer

# Data
from keras.datasets import mnist

# Others
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

np.random.seed(11111986)

# Load the MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
y_train = y_train.astype('int32')
X_test = X_test.astype('float32')
y_test = y_test.astype('int32')

Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

# Put 2 arrays on the border of the images to form a 32x32 shape
N = X_train.shape[0]
left0 = np.zeros((N, 2, 28))
right0 = np.zeros((N, 2, 28))
upper0 = np.zeros((N, 32, 2))
lower0 = np.zeros((N, 32, 2))

X_train = np.concatenate([left0, X_train, right0], axis=1)
X_train = np.concatenate([upper0, X_train, lower0], axis=2)

N = X_test.shape[0]
left0 = np.zeros((N, 2, 28))
right0 = np.zeros((N, 2, 28))
upper0 = np.zeros((N, 32, 2))
lower0 = np.zeros((N, 32, 2))

X_test = np.concatenate([left0, X_test, right0], axis=1)
X_test = np.concatenate([upper0, X_test, lower0], axis=2)

X_train /= 255.
X_test /= 255.

X_train = X_train[:, None, :, :]
X_test = X_test[:, None, :, :]

if False:  # if apply the imagegenerator
    valid_size = int(0.2*X_train.shape[0])
    valid_inds = np.random.choice(range(X_train.shape[0]), valid_size, False)
    X_valid = X_train[valid_inds]
    Y_valid = Y_train[valid_inds]

    tr_inds = np.setdiff1d(np.arange(X_train.shape[0]), valid_inds)
    X_train = X_train[tr_inds]
    Y_train = Y_train[tr_inds]

    train_gen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    train_gen.fit(X_train)

    valid_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        vertical_flip=False
    )
    valid_gen.fit(X_valid)


# Define the model
input = Input(shape=(1, 32, 32,))
h1 = TT_Layer(tt_input_shape=[4, 8, 8, 4], tt_output_shape=[4, 8, 8, 4], tt_ranks=[1, 3, 3, 3, 1],
              bias=True, activation='relu', kernel_regularizer=l2(5e-4), debug=False, ortho_init=True)(input)
# Alternatively, try a dense layer:
# h1 = Dense(output_dim=32*32, activation='relu', kernel_regularizer=l2(5e-4))(input)
output = Dense(output_dim=10, activation='softmax', kernel_regularizer=l2(1e-3))(h1)

model = Model(input=input, output=output)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# either the old fashion:
model.fit(x=X_train, y=Y_train, verbose=2, epochs=50, batch_size=128,
          validation_split=0.2)

# or with ImageDataGenerator
# model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=128),
#                     samples_per_epoch=len(X_train), nb_epoch=50, verbose=2,
#                     validation_data=valid_gen.flow(X_valid, Y_valid),
#                     nb_val_samples=X_valid.shape[0])


# Fitted values: AUROC/AUPRC/ACC
Y_hat = model.predict(x=X_train)
print roc_auc_score(Y_train, Y_hat)
print average_precision_score(Y_train, Y_hat)
print accuracy_score(Y_train, np.round(Y_hat))


# Predicted values:
Y_pred = model.predict(x=X_test)
print roc_auc_score(Y_test, Y_pred)
print average_precision_score(Y_test, Y_pred)
print accuracy_score(Y_test, np.round(Y_pred))
# 0.99970343541
# 0.997838863715
# 0.9785

# TT Layer compresses the first weight matrix to a factor of 1248 / 1048576 = 0.00119
# 9s per epoch
# Test error 0.0215 after 50 epochs, I think we can definitely train/tune the model further


# Without the TT Layer:

X_train = X_train.reshape((X_train.shape[0], 32*32))
X_test = X_test.reshape((X_test.shape[0], 32*32))

input = Input(shape=(32*32,))
h1 = Dense(output_dim=32*32, activation='relu', kernel_regularizer=l2(5e-4))(input)
output = Dense(output_dim=10, activation='softmax', kernel_regularizer=l2(1e-3))(h1)
model = Model(input=input, output=output)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=X_train, y=Y_train, verbose=2, nb_epoch=50, batch_size=128,
          validation_split=0.2)

# Fitted values: AUROC/AUPRC/ACC
Y_hat = model.predict(x=X_train)
print roc_auc_score(Y_train, Y_hat)
print average_precision_score(Y_train, Y_hat)
print accuracy_score(Y_train, np.round(Y_hat))


# Predicted values:
Y_pred = model.predict(x=X_test)
print roc_auc_score(Y_test, Y_pred)
print average_precision_score(Y_test, Y_pred)
print accuracy_score(Y_test, np.round(Y_pred))

# 0.999554701249
# 0.996718126202
# 0.9686

# ca 48s on average per epoch
# Test error 0.0313 after 50 epochs.






