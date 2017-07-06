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
A comparison between TT layer and dense layer is conducted using four gene expression-related datasets
from the R package spls:
    Dongjun Chung, Hyonho Chun and Sunduz Keles (2013).
    spls: Sparse Partial Least Squares (SPLS) Regression and Classification.
    R package version 2.2-1. https://CRAN.R-project.org/package=spls

For more background and citation information of the datasets please kindly refer the R documentation
    https://cran.r-project.org/web/packages/spls/spls.pdf

1) dataset 'prostate': binary classification
2) dataset 'lymphoma': 3-D classification
3) dataset 'mice': 83-D regression
4) dataset 'yeast': 18-D regression

The experimental results using current hyper parameters:
prostate data: -----------------------------------------

Results of the model with fully connected layer
Time consumed: 0:02:23.000157
Accuracy: 0.818181818182
AUROC: 0.941666666667
AUPRC: 0.952632552633

Results of the model with TT layer
Time consumed: 0:00:32.039866
Accuracy: 0.863636363636
AUROC: 0.958333333333
AUPRC: 0.963626419876

Compression factor = 22650 / 3937500 = 0.00575238095238

#-------------------------------------------------------

lymphoma: ----------------------------------------------

Results of the model with fully connected layer
Time consumed: 0:01:21.464509
Accuracy: 0.916666666667
AUROC: 1.0
AUPRC: 1.0

Results of the model with TT layer
Time consumed: 0:00:18.738026
Accuracy: 1.0
AUROC: 1.0
AUPRC: 1.0

Compression factor = 19200 / 2560000 = 0.0075
#-------------------------------------------------------

mice data: --------------------------------------------

Results of the model with fully connected layer
Time consumed: 0:01:13.798961
MSE: 0.230825076654

Results of the model with TT layer
Time consumed: 0:00:52.585227
MSE: 0.253088545206

Compression factor = 10500 / 506250 = 0.0207407407407
#-------------------------------------------------------

yeast data: --------------------------------------------
Results of the model with fully connected layer
Time consumed: 0:00:26.104242
MSE: 0.180198763472

Results of the model with TT layer
Time consumed: 0:00:18.263587
MSE: 0.184976958635

Compression factor = 3000 / 15000 = 0.2
#-------------------------------------------------------

There seems to be large margin of improvement in term of finer tuning the hyper parameters.

Insights, recommendations and critics are welcome and highly appreciated.
"""

# Basic
import sys
import numpy as np
from datetime import datetime

# Keras Model
from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

# TT Layer
from TTLayer import TT_Layer

# misc
from keras.utils.np_utils import to_categorical


np.random.seed(11111986)

# choose one data set:
data_name = 'prostate'
# data_name = 'lymphoma'
# data_name = 'mice'
# data_name = 'yeast'

data_path = './Datasets/GenExpress/'
X_train = np.loadtxt(data_path + data_name + '/' + data_name + '_train.data.gz')
Y_train = np.loadtxt(data_path + data_name + '/' + data_name + '_train.labels.gz')
X_valid = np.loadtxt(data_path + data_name + '/' + data_name + '_valid.data.gz')
Y_valid = np.loadtxt(data_path + data_name + '/' + data_name + '_valid.labels.gz')


n, d = X_train.shape
print 'Training data has shape = ' + str(X_train.shape)
print 'Valid data has shape = ' + str(X_valid.shape)


# Additional columns so that the TT can factorize the feature dimension
new_dim = None
if data_name == 'prostate':
    new_dim = 6300
elif data_name == 'lymphoma':
    new_dim = 4096
elif data_name == 'mice':
    new_dim = 150
elif data_name == 'yeast':
    new_dim = 120

X_train = np.concatenate([X_train, np.zeros((X_train.shape[0], new_dim - d))], axis=1)
col_shuffle = np.random.choice(range(new_dim), new_dim, False)
X_train = X_train[:, col_shuffle]
X_valid = np.concatenate([X_valid, np.zeros((X_valid.shape[0], new_dim - d))], axis=1)
X_valid = X_valid[:, col_shuffle]


normalization = 1
if normalization == 0:
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_train[np.where(np.isnan(X_train))] = 0.
    X_valid = (X_valid - X_valid.mean(axis=0)) / X_valid.std(axis=0)
    X_valid[np.where(np.isnan(X_valid))] = 0.
elif normalization == 1:
    X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
    X_valid = (X_valid - X_valid.min(axis=0)) / (X_valid.max(axis=0) - X_valid.min(axis=0))
    X_train[np.where(np.isnan(X_train))] = 0.
    X_valid[np.where(np.isnan(X_valid))] = 0.


if data_name == 'lymphoma':
    Y_train = to_categorical(Y_train.astype('int32'))
    Y_valid = to_categorical(Y_valid.astype('int32'))

# Hyper-parameters
if data_name == 'prostate':
    alpha = 0.1
    tt_alpha = 5e-4
    nb_epoch = 200
    batch_size = 10
    lr = 1e-4
    tt_input_shape = [7, 9, 10, 10]
    tt_output_shape = [5, 5, 5, 5]
    tt_ranks = [1, 15, 15, 15, 1]
elif data_name == 'lymphoma':
    alpha = 0.1
    tt_alpha = 5e-4
    nb_epoch = 200
    batch_size = 10
    lr = 1e-4
    tt_input_shape = [8, 8, 8, 8]
    tt_output_shape = [5, 5, 5, 5]
    tt_ranks = [1, 15, 15, 15, 1]
elif data_name == 'mice':
    alpha = 0.1
    tt_alpha = 5e-4
    nb_epoch = 100
    batch_size = 7
    lr = 1e-4
    tt_input_shape = [5, 6, 5]
    tt_output_shape = [15, 15, 15]
    tt_ranks = [1, 10, 10, 1]
elif data_name == 'yeast':
    alpha = 0.001
    tt_alpha = 0
    nb_epoch = 500
    batch_size = 5
    lr = 1e-4
    tt_input_shape = [4, 5, 6]
    tt_output_shape = [5, 5, 5]
    tt_ranks = [1, 10, 10, 1]


# Model with fully connected layer
if data_name == 'prostate':
    input = Input(shape=(new_dim,))
    h = Dense(output_dim=np.prod(tt_output_shape), activation='sigmoid', kernel_regularizer=l2(alpha))(input)
    output = Dense(output_dim=1, activation='sigmoid', kernel_regularizer=l2(alpha))(h)
    model_full = Model(input=input, output=output)
    model_full.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
elif data_name == 'lymphoma':
    input = Input(shape=(new_dim,))
    h = Dense(output_dim=np.prod(tt_output_shape), activation='sigmoid', kernel_regularizer=l2(alpha))(input)
    output = Dense(output_dim=Y_train.shape[1], activation='softmax', kernel_regularizer=l2(alpha))(h)
    model_full = Model(input=input, output=output)
    model_full.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
elif data_name in ['mice', 'yeast']:
    input = Input(shape=(new_dim,))
    h = Dense(output_dim=np.prod(tt_output_shape), activation='sigmoid', kernel_regularizer=l2(alpha))(input)
    output = Dense(output_dim=Y_train.shape[1], activation='linear', kernel_regularizer=l2(alpha))(h)
    model_full = Model(input=input, output=output)
    model_full.compile(optimizer=Adam(lr), loss='mse')

start_full = datetime.now()
for l in range(nb_epoch):
    if_print = l % 10 == 0
    if if_print:
        print 'iter = ' + str(l)
        verbose = 2
    else:
        verbose = 0
    history = model_full.fit(x=X_train, y=Y_train, verbose=verbose, epochs=1, batch_size=batch_size,
                             validation_split=0.2)
stop_full = datetime.now()


# Model with TT layer
if data_name == 'prostate':
    input_TT = Input(shape=(new_dim,))
    tt = TT_Layer(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, kernel_regularizer=l2(tt_alpha),
                  tt_ranks=tt_ranks, bias=True, activation='sigmoid', ortho_init=True)
    h_TT = tt(input_TT)
    # h_TT = Dropout(0.5)(h_TT)
    output_TT = Dense(output_dim=1, activation='sigmoid', kernel_regularizer=l2(alpha))(h_TT)
    model_TT = Model(input=input_TT, output=output_TT)
    model_TT.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
elif data_name == 'lymphoma':
    input_TT = Input(shape=(new_dim,))
    tt = TT_Layer(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, kernel_regularizer=l2(tt_alpha),
                  tt_ranks=tt_ranks, bias=True, activation='sigmoid', ortho_init=True)
    h_TT = tt(input_TT)
    output_TT = Dense(output_dim=Y_train.shape[1], activation='softmax', kernel_regularizer=l2(alpha))(h_TT)
    model_TT = Model(input=input_TT, output=output_TT)
    model_TT.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
elif data_name in ['mice', 'yeast']:
    input_TT = Input(shape=(new_dim,))
    tt = TT_Layer(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, kernel_regularizer=l2(tt_alpha),
                  tt_ranks=tt_ranks, bias=True, activation='sigmoid', ortho_init=True)
    h_TT = tt(input_TT)
    output_TT = Dense(output_dim=Y_train.shape[1], activation='linear', kernel_regularizer=l2(alpha))(h_TT)
    model_TT = Model(input=input_TT, output=output_TT)
    model_TT.compile(optimizer=Adam(lr), loss='mse')


start_TT = datetime.now()
for l in range(nb_epoch):
    if_print = l % 10 == 0
    if if_print:
        print 'iter = ' + str(l)
        verbose = 2
    else:
        verbose = 0
    history = model_TT.fit(x=X_train, y=Y_train, verbose=verbose, epochs=1, batch_size=batch_size,
                           validation_split=0.2)

stop_TT = datetime.now()


if data_name not in ['mice', 'yeast']:
    print '#######################################################'
    Y_pred_full = model_full.predict(X_valid)
    print 'Results of the model with fully connected layer'
    print 'Time consumed: ' + str(stop_full - start_full)
    print 'Accuracy: ' + str(accuracy_score(Y_valid, np.round(Y_pred_full)))
    print 'AUROC: ' + str(roc_auc_score(Y_valid, Y_pred_full))
    print 'AUPRC: ' + str(average_precision_score(Y_valid, Y_pred_full))
    print '#######################################################'
    Y_pred_TT = model_TT.predict(X_valid)
    print 'Results of the model with TT layer'
    print 'Time consumed: ' + str(stop_TT - start_TT)
    print 'Parameter compression factor: ' + str(tt.compress_factor)
    print 'Accuracy: ' + str(accuracy_score(Y_valid, np.round(Y_pred_TT)))
    print 'AUROC: ' + str(roc_auc_score(Y_valid, Y_pred_TT))
    print 'AUPRC: ' + str(average_precision_score(Y_valid, Y_pred_TT))
else:
    print '#######################################################'
    Y_pred_full = model_full.predict(X_valid)
    print 'Results of the model with fully connected layer'
    print 'Time consumed: ' + str(stop_full - start_full)
    print 'MSE: ' + str(((Y_valid - Y_pred_full)**2).mean())

    print '#######################################################'
    Y_pred_TT = model_TT.predict(X_valid)
    print 'Results of the model with TT layer'
    print 'Time consumed: ' + str(stop_TT - start_TT)
    print 'Parameter compression factor: ' + str(tt.compress_factor)
    print 'MSE: ' + str(((Y_valid - Y_pred_TT)**2).mean())
