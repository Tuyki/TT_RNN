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
This script would serve as a POC of applying (TT-)RNNs for sequence classification, as well as for some quick debugging 
  of the TT-RNN implementations. 
The main idea is to classify Fibonacci-, Lucas-, Padovan- and random sequences in their binary form (thus high-
  dimensional sequences). 
Out of each of the sequences, one samples at a random location a sub-sequence of random length. Such sub-sequences would
  funtion as the training and test data.
My experiments were performed on a Intel R Xeon R E7-4850 v2 2.30GHz, and the ACC and runtime are listed as below:  

RNN:      0.629   00:00:51
TT-RNN:   0.765   00:00:50
GRU:      0.893   00:01:48
TT-GRU:   0.975   00:01:22
LSTM:     0.925   00:02:41
TT-LSTM:  0.935   00:01:38

The same message as with the video data: TT-GRU seems to be the best
It's interesting to see that plain LSTM outperforms plain GRU.
But these results involve only one experiment setting afterall. 
"""

import numpy as np
import sys
import datetime
import time
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Dropout
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

from TTRNN import TT_RNN, TT_GRU, TT_LSTM

# Function that performs binary adding without transforming into decimal since even int64 would not suffice...
def bin_add(x,y):
    x = ''.join(x.astype('string'))
    y = ''.join(y.astype('string'))

    # https://stackoverflow.com/questions/21420447/need-help-in-adding-binary-numbers-in-python
    maxlen = max(len(x), len(y))

    #Normalize lengths
    x = x.zfill(maxlen)
    y = y.zfill(maxlen)

    result = ''
    carry = 0

    for i in range(maxlen-1, -1, -1):
        r = carry
        r += 1 if x[i] == '1' else 0
        r += 1 if y[i] == '1' else 0

        result = ('1' if r % 2 == 1 else '0') + result
        carry = 0 if r < 2 else 1

    if carry !=0 : result = '1' + result

    ret = result.zfill(maxlen)
    return np.array(list(ret)).astype('uint8')

# Function that generates fibo/luc/pad/random sequences in binary form
def gen_bin_seq(n=100, maxdim=512, type='fibo'):
    ret = np.zeros((n, maxdim), dtype='uint8')

    if type == 'fibo':
        ret[0, -1] = 1
        ret[1, -1] = 1
        for i in range(2, n):
            ret[i] = bin_add(ret[i-1], ret[i-2])
    elif type == 'lucas':
        ret[0, -2] = 1
        ret[0, -1] = 0
        ret[1, -1] = 1
        for i in range(2, n):
            ret[i] = bin_add(ret[i-1], ret[i-2])
    elif type == 'padovan':
        ret[0, -1] = 1
        ret[1, -1] = 1
        ret[2, -1] = 1
        for i in range(3, n):
            ret[i] = bin_add(ret[i-2], ret[i-3])
    elif type == 'random':
        ret = np.random.choice([0, 1], n*maxdim, True).reshape((n, maxdim)).astype('uint8')
    return ret


np.random.seed(11111986)

# Experiment settings
model_type = 0  # 0 for RNN, 1 for GRU, 2 for LSTM
use_TT = 1      # 0 for non-TT, 1 for TT
rank = 3        # tt ranks


# Generate data
seq_length = 400  # the first k fib/luc/pad/rand numbers
fib_bin = gen_bin_seq(seq_length, maxdim=512, type='fibo')
luc_bin = gen_bin_seq(seq_length, maxdim=512, type='lucas')
pad_bin = gen_bin_seq(seq_length, maxdim=512, type='padovan')
rand_bin = gen_bin_seq(seq_length, maxdim=512, type='random')

N = 300  # number of training samples
T = 30   # number of time steps
d = fib_bin.shape[1]

X_fib = [None]*N
X_luc = [None]*N
X_pad = [None]*N
X_ran = [None]*N

for i in range(N):
    this_len = np.random.choice(range(3, T+1), 1)[0]
    this_start = np.random.choice(range(0, seq_length-this_len), 1)[0]
    X_fib[i] = fib_bin[this_start:this_start+this_len]

    this_len = np.random.choice(range(3, T+1), 1)[0]
    this_start = np.random.choice(range(0, seq_length - this_len), 1)[0]
    X_luc[i] = luc_bin[this_start:this_start+this_len]

    this_len = np.random.choice(range(3, T+1), 1)[0]
    this_start = np.random.choice(range(0, seq_length - this_len), 1)[0]
    X_pad[i] = pad_bin[this_start:this_start + this_len]

    this_len = np.random.choice(range(3, T + 1), 1)[0]
    this_start = np.random.choice(range(0, seq_length - this_len), 1)[0]
    X_ran[i] = rand_bin[this_start:this_start + this_len]

X_fib = pad_sequences(X_fib)
X_luc = pad_sequences(X_luc)
X_pad = pad_sequences(X_pad)
X_ran = pad_sequences(X_ran)

X = np.concatenate([X_fib, X_luc, X_pad, X_ran], 0)
Y = np.concatenate([np.repeat([0], N), np.repeat([1], N), np.repeat([2], N), np.repeat([3], N)], )
Y = to_categorical(Y)

r = rank
tt_input_shape = [16, 32]
tt_output_shape = [8, 8]
tt_ranks = [1, r, 1]

input = Input(shape=(T, d))
if model_type == 0:
    if use_TT:
        rnn_layer = TT_RNN(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, tt_ranks=tt_ranks,
                           return_sequences=False, debug=True,
                           dropout=.25, recurrent_dropout = .25, activation='tanh', )
    else:
        rnn_layer = SimpleRNN(units=np.prod(tt_output_shape),
                              return_sequences=False,
                              dropout=.25, recurrent_dropout=.25, activation='tanh',)
elif model_type == 1:
    if use_TT:
        rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, tt_ranks=tt_ranks,
                           return_sequences=False, debug=True,
                           dropout=.25, recurrent_dropout=.25, activation='tanh', )
    else:
        rnn_layer = GRU(units=np.prod(tt_output_shape),
                        return_sequences=False,
                        dropout=.25, recurrent_dropout=.25, activation='tanh', )
elif model_type == 2:
    if use_TT:
        rnn_layer = TT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, tt_ranks=tt_ranks,
                            return_sequences=False, debug=True,
                            dropout=.25, recurrent_dropout=.25, activation='tanh', )
    else:
        rnn_layer = LSTM(units=np.prod(tt_output_shape),
                         return_sequences=False,
                         dropout=.25, recurrent_dropout=.25, activation='tanh', )

h = rnn_layer(input)
output = Dense(units=4, activation='softmax')(h)
model = Model(input, output)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['categorical_accuracy'])  #

shuffle = np.random.choice(range(X.shape[0]), X.shape[0], False)
X = X[shuffle]
Y = Y[shuffle]

X_train = X[0:(3*N/2)]
Y_train = Y[0:(3*N/2)]

X_test = X[(3*N/2)::]
Y_test = Y[(3*N/2)::]

maxIter = 100
test_acc = np.zeros((maxIter,))

# just to have the model compiled in advance
model.fit(X_train, Y_train, epochs=1, batch_size=16, verbose=2, validation_data=[X_test, Y_test])

start = datetime.datetime.now()
for l in range(maxIter):
    print l
    history = model.fit(X_train, Y_train, epochs=1, batch_size=16, verbose=2, validation_data=[X_test, Y_test])
    test_acc[l] = history.history['val_categorical_accuracy'][0]
stop = datetime.datetime.now()

print stop - start
print model.evaluate(X_test, Y_test)
