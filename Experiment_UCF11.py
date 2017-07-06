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


def get_clips(class_name):
    files = os.listdir(data_path + class_name)
    files.sort()
    clip_list = []
    for this_file in files:
        clips = os.listdir(data_path + class_name + '/' + this_file)
        clips.sort()
        for this_clip in clips:
            clip_list.append( data_path + class_name + '/' + this_file + '/' + this_clip )
    return clip_list


def load_data(inds, mode = 'train'):
    N = len(inds)
    X = np.zeros((N, GLOBAL_MAX_LEN, 120*160*3), dtype='int8')
    if mode == 'train':
        set = train_set
    else:
        set = test_set
    for i in range(N):
        read_in = open(set[0][inds[i]])
        this_clip = pickle.load(read_in)[0] # of shape (nb_frames, 240, 320, 3)
        read_in.close()
        # flatten the dimensions 1, 2 and 3
        this_clip = this_clip.reshape(this_clip.shape[0], -1) # of shape (nb_frames, 240*320*3)
        this_clip = (this_clip - 128.).astype('int8')   # this_clip.mean()
        X[i] = pad_sequences([this_clip], maxlen=GLOBAL_MAX_LEN, truncating='post', dtype='int8')[0]
    Y = set[1][inds]
    return [X, Y]


# Load the data --------------------------------------------------------------------------------------------------------
np.random.seed(11111986)

# Settings:

CV_setting = 0  # [0, 1, 2, 3, 4]
model_type = 1  # 0 for GRU, 1 for LSTM
use_TT = 0      # 0 for non-TT, 1 for TT

# Had to remove due to anonymity
data_path = ''
write_out_path = ''

classes = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding', 'soccer_juggling',
           'swing', 'tennis_swing', 'trampoline_jumping', 'volleyball_spiking', 'walking']

clips = [None]*11
labels = [None]*11
sizes = np.zeros(11)
for k in range(11):
    this_clip = get_clips(classes[k])
    clips[k] = this_clip
    sizes[k] = len(this_clip)
    labels[k] = np.repeat([k], sizes[k])

# flatten both lists
clips = np.array( [item for sublist in clips for item in sublist] )
labels = np.array( [item for sublist in labels for item in sublist] )
labels = to_categorical(labels)

# iterate through all clips and store the length of each:
if False: # first run
    lengths = np.zeros(len(clips))
    for l in range(len(clips)):
        read_in = open(clips[l], 'r')
        this_clip = pickle.load(read_in)[0]
        read_in.close()
        lengths[l] = this_clip.shape[0]


GLOBAL_MAX_LEN = 1492

shuffle_inds = np.random.choice(range(len(clips)), len(clips), False)
clips = clips[shuffle_inds]
labels = labels[shuffle_inds]

CV_splits = np.array_split(np.arange(len(clips)), 5)

test_inds = CV_splits[CV_setting]
train_inds = np.setdiff1d(np.arange(len(clips)), test_inds)


train_set = [clips[train_inds], labels[train_inds]]
test_set = [clips[test_inds], labels[test_inds]]

n_tr = len(train_set[0])
n_te = len(test_set[0])

# X_train, Y_train = load_data(np.arange(0, 128), mode='train')  # small set
X_train, Y_train = load_data(np.arange(0, n_tr), mode='train')  # full set
# X_test, Y_test = load_data(np.arange(0, 128), mode='test')  # small set
X_test, Y_test = load_data(np.arange(0, n_te), mode='test')  # full set


# Define the model -----------------------------------------------------------------------------------------------------
tt_input_shape = [8, 20, 20, 18]
tt_output_shape = [4, 4, 4, 4]
tt_ranks = [1, 4, 4, 4, 1]
alpha = 1e-2

input = Input(shape=(GLOBAL_MAX_LEN, 120*160*3))
masked_input = Masking(mask_value=0, input_shape=(GLOBAL_MAX_LEN, 120*160*3))(input)
if model_type == 0:
    if use_TT == 0:
        rnn_layer = GRU(output_dim=np.array(tt_output_shape).prod(),
                        return_sequences=False,
                        dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    else:
        rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                           tt_ranks=tt_ranks,
                           return_sequences=False,
                           dropout=0.25, recurrent_dropout=0.25, activation='tanh')
else:
    if use_TT == 0:
        rnn_layer = LSTM(output_dim=np.array(tt_output_shape).prod(),
                         return_sequences=False,
                         dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    else:
        rnn_layer = TT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                           tt_ranks=tt_ranks,
                            return_sequences=False,
                            dropout=0.25, recurrent_dropout=0.25, activation='tanh')
h = rnn_layer(masked_input)
output = Dense(output_dim=11, activation='softmax', W_regularizer=l2(alpha))(h)
model = Model(input, output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Start training -------------------------------------------------------------------------------------------------------
start = datetime.datetime.now()
for l in range(1001):
    print 'iter ' + str(l)
    model.fit(X_train, Y_train, nb_epoch=1, batch_size=16, verbose=1, validation_data=[X_test, Y_test])

    # if l % 10 == 0:
    #     save_name = str(CV_setting) + '_' + str(model_type) + '_' + str(use_TT)
    #     write_out = open(write_out_path + save_name +'.pkl', 'wb')
    #     pickle.dump(model.get_weights(), write_out)
    #     write_out.close()
stop = datetime.datetime.now()
print model.evaluate(X_test, Y_test)


compress_factor = 1.
if use_TT == 1:
    compress_factor = rnn_layer.compress_factor
