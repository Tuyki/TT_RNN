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

data_path = ''
write_out_path = ''
splits_path = ''

def get_clips(class_name):
    files = os.listdir(data_path + class_name)
    files.sort()
    clip_list = []
    for this_file in files:
        clip_list.append( data_path + class_name + '/' + this_file )
    return clip_list


np.random.seed(11111986)

CV_setting = 1  # [1, 2, 3]
model_type = 0
use_TT = 1

classes = os.listdir(data_path)
classes.sort()



clips = [None]*51
labels = [None]*51
sizes = np.zeros(51)
X_train_lists = [None]*51
splits = [None]*51
for k in range(51):
    this_clip = get_clips(classes[k])
    clips[k] = this_clip
    sizes[k] = len(this_clip)
    labels[k] = np.repeat([k], sizes[k])
    split_read = splits_path + classes[k] + '_split.txt'
    splits[k] = np.genfromtxt(split_read, dtype=None)[:, CV_setting]

# flatten both lists
clips = np.array( [item for sublist in clips for item in sublist] )
splits = np.array( [item for sublist in splits for item in sublist] )
labels = np.array( [item for sublist in labels for item in sublist] )
labels = to_categorical(labels)

GLOBAL_MAX_LEN = 90


np.random.seed(11111986)
shuffle_inds = np.random.choice(range(len(clips)), len(clips), False)
clips = clips[shuffle_inds]
labels = labels[shuffle_inds]

CV_splits = np.array_split(np.arange(len(clips)), 5)

test_inds = CV_splits[CV_setting]
train_inds = np.setdiff1d(np.arange(len(clips)), test_inds)

train_set = [clips[train_inds], labels[train_inds]]
test_set = [clips[test_inds], labels[test_inds]]


def load_data(inds, mode = 'train', maxlen = GLOBAL_MAX_LEN):
    N = len(inds)
    X = np.zeros((N, maxlen, 120*160*3), dtype='int8')
    if mode == 'train':
        set = train_set
    else:
        set = test_set
    for i in range(N):
        read_in = open(set[0][inds[i]])
        this_clip = pickle.load(read_in)[0]  # of shape (nb_frames, 240, 320, 3)
        read_in.close()
        # take every 3rd frame due to memory restriction
        this_clip = this_clip[np.arange(0, this_clip.shape[0], 3)] # of shape (nb_frames/3, 240, 320, 3)
        # flatten the dimensions 1, 2 and 3
        this_clip = this_clip.reshape(this_clip.shape[0], -1) # of shape (nb_frames/3, 240*320*3)
        this_clip = (this_clip - 128).astype('int8')  # this_clip.mean()
        X[i] = pad_sequences([this_clip], maxlen=maxlen, truncating='post', dtype='int8')[0]
    Y = set[1][inds]

    return [X, Y]


n_tr = len(train_set[0])
n_te = len(test_set[0])



# test
# X_train, Y_train = load_data(np.arange(500),mode='train')
# X_test, Y_test = load_data(np.arange(100),mode='test')

alpha = 1e-2

tt_input_shape = [4, 20, 20, 36]
tt_output_shape = [4, 4, 4, 4]
tt_ranks = [1, 4, 4, 4, 1]


input = Input(shape=(GLOBAL_MAX_LEN, 120*160*3))

if model_type == 0:
    if use_TT == 0:
        rnn_layer = GRU(output_dim=225, activation='tanh', 
                        dropout=0.25, recurrent_dropout=0.25, )
    elif use_TT == 1:
        rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                           tt_ranks=tt_ranks,
                           dropout=0.25, recurrent_dropout=0.25, activation='tanh',  
                           return_sequences=False, ortho_init=False )
elif model_type == 1:
    if use_TT == 0:
        rnn_layer = LSTM(output_dim=225, activation='tanh', consume_less='gpu',
                         dropout=0.25, recurrent_dropout=0.25, )
    elif use_TT == 1:
        rnn_layer = TT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                            tt_ranks=tt_ranks,
                            dropout=0.25, recurrent_dropout=0.25, activation='tanh',
                            return_sequences=False, ortho_init=False)
h = rnn_layer(input)
h = Dropout(0.25)(h)
output = Dense(output_dim=51, activation='softmax', kernel_regularizer=l2(alpha))(h)
model = Model(input, output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


if use_TT == 1:
    compress_factor = rnn_layer.compress_factor
else:
    compress_factor = 1

# X_train, Y_train = load_data(np.arange(0, 16), mode='train')
X_train, Y_train = load_data(np.arange(0, n_tr), mode='train')
# X_test, Y_test = load_data(np.arange(0, 16), mode='test')
X_test, Y_test = load_data(np.arange(0, n_te), mode='test')

file_name = str(CV_setting) + '_' + str(model_type) + '_' + str(use_TT)
start = datetime.datetime.now()
for l in range(401):
    print 'iter ' + str(l)
    model.fit(X_train, Y_train, epochs=1, batch_size=16, verbose=1, )
    if l % 50 == 0:
        res = model.evaluate(X_test, Y_test)
        print 'Test res: '
        print res
        write_out = open(write_out_path + file_name + '_' + str(l) + '.pkl', 'wb')
        pickle.dump([model.get_weights(), res, compress_factor], write_out)
        write_out.close()

stop = datetime.datetime.now()
print stop-start
