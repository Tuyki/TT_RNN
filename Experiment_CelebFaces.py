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


import sys
import datetime
import time

import os
import numpy as np
import pickle

from keras.layers import Input, GRU, LSTM, Dense, Dropout, Masking, BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score

# Custom Functions -----------------------------------------------------------------------------------------------------

from TTRNN import TT_GRU, TT_LSTM


def get_clips(class_name):
    files = os.listdir(data_path + class_name)
    files.sort()
    clip_list = []
    for this_file in files:
        clip_list.append( data_path + class_name + '/' + this_file )
    return clip_list


def load_data(inds, mode = 'train', maxlen = 85):
    N = len(inds)
    X = np.zeros((N, maxlen, 120*160*3), dtype='int8')
    if mode == 'train':
        set = train_set
    else:
        set = test_set
    for i in range(N):
        read_in = open(set[0][inds[i]])
        this_clip = pickle.load(read_in)[0]
        read_in.close()
        this_clip = this_clip.reshape(this_clip.shape[0], -1)
        this_clip = (this_clip - 128).astype('int8')
        X[i] = pad_sequences([this_clip], maxlen=maxlen, truncating='post', dtype='int8')[0]
    Y = set[1][inds]

    return [X, Y]


# Load the data --------------------------------------------------------------------------------------------------------
np.random.seed(11111986)

# Settings:
# CV_setting = int(sys.argv[1])
# model_type= int(sys.argv[2])  # [0, 1] for GRU, LSTM
# use_TT = int(sys.argv[3])  # [0, 1] for False, True

CV_setting = 0
model_type = 1
use_TT = 0

# Had to remove due to anonymity
data_path = ''
write_out_path = ''

files = os.listdir(data_path)
files.sort()

N = len(files)
targets = ['']*N
for l in range(N):
    # l = 0
    this_file = files[l]
    this_file_split = this_file.split('_')
    this_file_split[-1] = this_file_split[-1].split('.')[0]
    targets[l] = this_file_split[-2] + '_' + this_file_split[-1]

targets = np.array(targets)

classes = np.unique(targets)

Y = np.zeros(N, dtype='int8')
for l in range(N):
    Y[l] = np.where(classes == targets[l])[0]
Y = to_categorical(Y).astype('int8')


GLOBAL_MAX_LEN=85

clips = np.array([data_path + this_file for this_file in files])
shuffle_ind = np.random.choice(range(N), N, False)
clips = clips[shuffle_ind]
Y = Y[shuffle_ind]

CV_splits = np.array_split(np.arange(N), 5)

test_inds = CV_splits[CV_setting]
train_inds = np.setdiff1d(np.arange(N), test_inds)

train_set = [clips[train_inds], Y[train_inds]]
test_set = [clips[test_inds], Y[test_inds]]


n_tr = len(train_set[0])
n_te = len(test_set[0])

# X_train, Y_train = load_data(np.arange(0, 512), mode='train')
X_train, Y_train = load_data(np.arange(0, n_tr), mode='train')
# X_test, Y_test = load_data(np.arange(0, 256), mode='test')
X_test, Y_test = load_data(np.arange(0, n_te), mode='test')


# Define the model -----------------------------------------------------------------------------------------------------
dropoutRate = 0.25
alpha = 1e-2

tt_input_shape = [4, 20, 20, 36]
tt_output_shape = [4, 4, 4, 4]
tt_ranks = [1, 4, 4, 4, 1]

input = Input(shape=(GLOBAL_MAX_LEN, 120*160*3))
if model_type == 0:
    if use_TT == 0:
        rnn_layer = GRU(output_dim=225,
                        return_sequences=False,
                        dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    elif use_TT == 1:
        rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                           tt_ranks=tt_ranks,
                           return_sequences=False,
                           dropout=0.25, recurrent_dropout=0.25, activation='tanh')
else:
    if use_TT == 0:
        rnn_layer = LSTM(output_dim=225,
                         return_sequences=False,
                         dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    elif use_TT == 1:
        rnn_layer = TT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                            tt_ranks=tt_ranks,
                            return_sequences=False,
                            dropout=0.25, recurrent_dropout=0.25, activation='tanh')
h = rnn_layer(input)
output = Dense(output_dim=47, activation='softmax', kernel_regularizer=l2(alpha))(h)
model = Model(input, output)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


if use_TT == 1:
    compress_factor = rnn_layer.compress_factor
else:
    compress_factor = 1


print n_tr
print n_te


# Start training -------------------------------------------------------------------------------------------------------
if True:

    file_name = str(CV_setting) + '_' + str(model_type) + '_' + str(use_TT)
    start = datetime.datetime.now()

    for l in range(101):
        print 'iter ' + str(l)
        model.fit(X_train, Y_train, nb_epoch=1, batch_size=16, verbose=1, validation_data=[X_test, Y_test])

    stop = datetime.datetime.now()
    print stop-start

    res = model.evaluate(X_test, Y_test)
    save_name = str(CV_setting) +'_'+ str(model_type) + '_' + str(use_TT)
    write_out = open(write_out_path + save_name + '.pkl', 'wb')
    pickle.dump([model.get_weights(), res, stop-start], write_out)
    write_out.close()
