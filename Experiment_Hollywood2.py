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

import os
import numpy as np
import pickle
import datetime

from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, Masking, BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

# Custom Functions -----------------------------------------------------------------------------------------------------
from TTRNN import TT_GRU, TT_LSTM


def load_data(inds, mode='train'):
    N = len(inds)
    X = np.zeros((N, GLOBAL_MAX_LEN, 234*100*3), dtype='int8')
    Y = np.zeros((N, 12), dtype='int8')

    for i in range(N):
        print i
        if mode=='train':
            read_in = open(data_path + 'actioncliptrain/' + tr_sample_filenames[inds[i]])
        elif mode == 'test':
            read_in = open(data_path + 'actioncliptest/' + te_sample_filenames[inds[i]])
        this_clip = pickle.load(read_in)[0]
        read_in.close()
        # flatten the dimensions 1, 2 and 3
        this_clip = this_clip.reshape(this_clip.shape[0], -1) # of shape (nb_frames, 240*320*3)
        this_clip = (this_clip - 128).astype('int8')   # this_clip.mean()
        X[i] = pad_sequences([this_clip], maxlen=GLOBAL_MAX_LEN, truncating='post', dtype='int8')[0]
        if mode == 'train':
            Y[i] = tr_labels[inds[i]]
        elif mode == 'test':
            Y[i] = te_labels[inds[i]]
    return [X, Y]


# Load the data --------------------------------------------------------------------------------------------------------
np.random.seed(11111986)

# Settings:
model_type = 1
use_TT = 0


# Had to remove due to anonymity
data_path = ''
write_out_path = ''

GLOBAL_MAX_LEN = 1496

classes = ['AnswerPhone', 'DriveCar', 'Eat', 'FightPerson', 'GetOutCar', 'HandShake',
           'HugPerson', 'Kiss', 'Run', 'SitDown', 'SitUp', 'StandUp']


# filter out labels that are not included because their samples are longer than 50 frames:
tr_sample_filenames = os.listdir(data_path + 'actioncliptrain/')
tr_sample_filenames.sort()

tr_sample_ids = np.array( [x[-10:-5] for x in tr_sample_filenames] ).astype('int16')

tr_label_filename = data_path + 'labels_train.txt'
tr_labels = np.loadtxt(tr_label_filename, dtype='int16')
tr_label_ids = tr_labels[:, 0]
tr_labels = tr_labels[:, 1::]


# filter out labels that are not included because their samples are longer than 50 frames:
te_sample_filenames = os.listdir(data_path + 'actioncliptest/')
te_sample_filenames.sort()

te_sample_ids = np.array( [x[-10:-5] for x in te_sample_filenames] ).astype('int16')

te_label_filename = data_path + 'labels_test.txt'
te_labels = np.loadtxt(te_label_filename, dtype='int16')
te_label_ids = te_labels[:, 0]
te_labels = te_labels[:, 1::]

n_tr = tr_labels.shape[0]
n_te = te_labels.shape[0]


# X_train, Y_train = load_data(np.arange(0, 128), mode='train')  # small set
X_train, Y_train = load_data(np.arange(0, n_tr), mode='train')
# X_test, Y_test = load_data(np.arange(0, 128), mode='test')  # small set
X_test, Y_test = load_data(np.arange(0, n_te), mode='test')


# Define the model -----------------------------------------------------------------------------------------------------
alpha = 1e-2

tt_input_shape = [10, 18, 13, 30]
tt_output_shape = [4, 4, 4, 4]
tt_ranks = [1, 4, 4, 4, 1]

dropoutRate = .25

input = Input(shape=(GLOBAL_MAX_LEN, 234*100*3))
if model_type == 0:
    if use_TT ==0:
        rnn_layer = GRU(np.prod(tt_output_shape),
                        return_sequences=False,
                        dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    else:
        rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                           tt_ranks=tt_ranks,
                           return_sequences=False,
                           dropout=0.25, recurrent_dropout=0.25, activation='tanh')
else:
    if use_TT ==0:
        rnn_layer = LSTM(np.prod(tt_output_shape),
                         return_sequences=False,
                         dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    else:
        rnn_layer = TT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                            tt_ranks=tt_ranks,
                            return_sequences=False,
                            dropout=0.25, recurrent_dropout=0.25, activation='tanh')
h = rnn_layer(input)
output = Dense(units=12, activation='sigmoid', kernel_regularizer=l2(alpha))(h)
model = Model(input, output)
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')


# Start training -------------------------------------------------------------------------------------------------------
for l in range(501):
    print 'iter ' + str(l)
    model.fit(X_train, Y_train, nb_epoch=1, batch_size=32, verbose=1, validation_split=.15)

    if l % 10 == 0:
        Y_hat = model.predict(X_train)
        Y_pred = model.predict(X_test)
        train_res = average_precision_score(Y_train, Y_hat)
        test_res = average_precision_score(Y_test, Y_pred)

        print 'Training: '
        print train_res
        print 'Test: '
        print test_res
