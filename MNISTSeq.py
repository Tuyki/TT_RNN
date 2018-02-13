__author__ = "Yinchong Yang"
__copyright__ = "Siemens AG, 2018"
__licencse__ = "MIT"
__version__ = "0.1"

"""
MIT License
Copyright (c) 2018 Siemens AG
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
We first sample MNIST digits to form sequences of random lengths. 
The sequence is labeled as one if it contains a zero, and is labeled zero otherwise.
This simulates a high dimensional sequence classification task, such as predicting therapy decision
and survival of patients based on their historical clinical event information.    
We train plain LSTM and Tensor-Train LSTM for this task. 
After the training, we apply Layer-wise Relevance Propagation to identify the digit(s) that 
have influenced the classification. 
Apparently, we would expect the LRP algorithm would assign high relevance value to the zero(s)
in the sequence. 
These experiments turn out to be successful, which demonstrates that 
i) the LSTM and TT-LSTM can indeed learn the mapping from a zero to the sequence class, and that 
ii) both LSTMs have no problem in storing the zero pattern over a period of time, because the 
classifier is deployed only at the last hidden state, and that   
iii) the implementation of the LRP algorithm, complex as it is, is also correct, in that 
the zeros are assigned high relevance scores. 

Especially the experiments with the plain LSTM serve as simulation study supporting our submission of 
“Yinchong Yang, Volker Tresp, Marius Wunderle, Peter A. Fasching, 
Explaining Therapy Predictions with Layer-wise Relevance Propagation in Neural Networks, at IEEE ICHI 2018”. 

The original LRP for LSTM from the repository: 
                        https://github.com/ArrasL/LRP_for_LSTM
which we modified and adjusted for keras models.  

Feel free to experiment with the hyper parameters and suggest other sequence classification tasks.
Have fun ;)  
"""


import pickle
import sys
import numpy as np
from numpy import newaxis as na
import keras
from keras.layers.recurrent import Recurrent
from keras import backend as K
from keras.engine import InputSpec
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer

from TTLayer import *
from TTRNN import TT_LSTM


def make_seq(n, x, y, maxlen=32, seed=123):
    np.random.seed(seed)
    lens = np.random.choice(range(2, maxlen), n)
    seqs = np.zeros((n, maxlen, 28**2))
    labels = np.zeros(n)
    digits_label = np.zeros((n, maxlen), dtype='int32')-1
    ids = np.zeros((n, maxlen), dtype='int64')-1
    for i in range(n):
        digits_inds = np.random.choice(range(x.shape[0]), lens[i])
        ids[i, -lens[i]::] = digits_inds
        seqs[i, -lens[i]::, :] = x[digits_inds]
        digits_label[i, -lens[i]::] = y[digits_inds]
        class_inds = y[digits_inds]

        if True:
            # option 1: is there any 0 in the sequence?
            labels[i] = (0 in class_inds)
        else:
            # option 2: even number of 0 -> label=0, odd number of 0 -> label=1
            labels[i] = len(np.where(class_inds == 0)[0]) % 2 == 1
    return [seqs, labels, digits_label, ids]


# From: https://github.com/ArrasL/LRP_for_LSTM
def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False):
    """
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  number of lower-layer units onto which the bias/stabilizer contribution is redistributed
    - eps:            stabilizer (small positive number)
    - bias_factor:    for global relevance conservation set to 1.0, otherwise 0.0 to ignore bias redistribution
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """
    sign_out = np.where(hout[na, :] >= 0, 1., -1.)  # shape (1, M)

    numer = (w * hin[:, na]) + \
            ((bias_factor * b[na, :] * 1. + eps * sign_out * 1.) * 1. / bias_nb_units)  # shape (D, M)

    denom = hout[na, :] + (eps * sign_out * 1.)  # shape (1, M)

    message = (numer / denom) * Rout[na, :]  # shape (D, M)

    Rin = message.sum(axis=1)  # shape (D,)

    # Note: local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D
    #       global network relevance conservation if bias_factor==1.0 (can be used for sanity check)
    if debug:
        print("local diff: ", Rout.sum() - Rin.sum())

    return Rin


def sigmoid(x):
    x = x.astype('float128')
    return 1. / (1. + np.exp(-x))

# Modified from https://github.com/ArrasL/LRP_for_LSTM 
def lstm_lrp(l, d, train_data = True):
    if train_data:
        x_l = X_tr[l]
        y_l = Y_tr[l]
        z_l = Z_tr[l]
        # d_l = d_tr[l]
    else:
        x_l = X_te[l]
        y_l = Y_te[l]
        z_l = Z_te[l]
        # d_l = d_te[l]


    # calculate the FF pass in LSTM for every time step
    pre_gates = np.zeros((MAXLEN, d*4))
    gates = np.zeros((MAXLEN, d * 4))
    h = np.zeros((MAXLEN, d))
    c = np.zeros((MAXLEN, d))

    for t in range(MAXLEN):
        z = np.dot(x_l[t], Ws)
        if t > 0:
            z += np.dot(h[t-1], Us)
        z += b
        pre_gates[t] = z
        z0 = z[0:d]
        z1 = z[d:2*d]
        z2 = z[2*d:3*d]
        z3 = z[3 * d::]
        i = sigmoid(z0)
        f = sigmoid(z1)
        c[t] = f * c[t-1] + i * np.tanh(z2)
        o = sigmoid(z3)
        h[t] = o * np.tanh(c[t])
        gates[t] = np.concatenate([i, f, np.tanh(z2), o])

    # check: z_l[12] / h[-1][12]

    Rh = np.zeros((MAXLEN, d))
    Rc = np.zeros((MAXLEN, d))
    Rg = np.zeros((MAXLEN, d))
    Rx = np.zeros((MAXLEN, 28**2))

    bias_factor = 0

    Rh[MAXLEN-1] = lrp_linear(hin=z_l,
                              w=Dense_w,
                              b=np.array(Dense_b),
                              hout=np.dot(z_l, Dense_w)+Dense_b,
                              Rout=np.array([y_l]),
                              bias_nb_units=len(z_l),
                              eps=eps,
                              bias_factor=bias_factor)


    for t in reversed(range(MAXLEN)):
        # t = MAXLEN-1
        # print t

        Rc[t] += Rh[t]
        # Rc[t] = Rh[t]
        if t > 0:
            Rc[t-1] = lrp_linear(gates[t, d: 2 * d] * c[t - 1],  # gates[t , 2 *d: 3 *d ] *c[ t -1],
                                  np.identity(d),
                                  np.zeros((d)),
                                  c[t],
                                  Rc[t],
                                  2*d,
                                  eps,
                                  bias_factor,
                                  debug=False)

        Rg[t] = lrp_linear(gates[t, 0:d] * gates[t, 2*d:3*d],  # h_input: i + g
                           np.identity(d),                     # W
                           np.zeros((d)),                      # b
                           c[t],                               # h_output
                           Rc[t],                              # R_output
                           2 * d,
                           eps,
                           bias_factor,
                           debug=False)

        # foo = np.dot(x_l[t], Ws[:,2*d:3*d]) + np.dot(h[t-1], Us[:, 2*d:3*d]) + b[2*d:3*d]

        Rx[t] = lrp_linear(x_l[t],
                           Ws[:,2*d:3*d],
                           b[2*d:3*d],
                           pre_gates[t, 2*d:3*d],
                           Rg[t],
                           d + 28 ** 2,
                           eps,
                           bias_factor,
                           debug=False)

        if t > 0:
            Rh[t-1] = lrp_linear(h[t-1],
                                 Us[:,2*d:3*d],
                                 b[2*d:3*d],
                                 pre_gates[t, 2 * d:3 * d],
                                 Rg[t],
                                 d + 28**2,
                                 eps,
                                 bias_factor,
                                 debug=False)

    # hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False

    # Rx[np.where(d_l==-1.)[0]] *= 0
    return Rx


from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import Dense, GRU, LSTM, Dropout, Masking
from keras.optimizers import *
from keras.regularizers import l2

from sklearn.metrics import *


# Script configurations ###################################################################

seed=111111
use_TT = True  # whether use Tensor-Train or plain RNNs


# Prepare the data ########################################################################
# Load the MNIST data and build sequences:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

MAXLEN = 32  # max length of the sequences

X_tr, Y_tr, d_tr, idx_tr = make_seq(n=10000, x=x_train, y=y_train, maxlen=MAXLEN, seed=seed)
X_te, Y_te, d_te, idx_te = make_seq(n=1000, x=x_test, y=y_test, maxlen=MAXLEN, seed=seed+1)

# Define the model ######################################################################

if use_TT:
    # TT settings
    tt_input_shape = [7, 7, 16]
    tt_output_shape = [4, 4, 4]
    tt_ranks = [1, 4, 4, 1]

rnn_size = 64

X = Input(shape=X_tr.shape[1::])
X_mask = Masking(mask_value=0.0, input_shape=X_tr.shape[1::])(X)

if use_TT:
    Z = TT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, tt_ranks=tt_ranks,
                return_sequences=False, recurrent_dropout=.5)(X_mask)
    Out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(1e-2))(Z)
else:
    Z = LSTM(units=rnn_size, return_sequences=False, recurrent_dropout=.5)(X_mask) # dropout=.5,
    Out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(1e-2))(Z)

rnn_model = Model(X, Out)
rnn_model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the model and save the results ######################################################
rnn_model.fit(X_tr, Y_tr, epochs=50, batch_size=32, validation_split=.2, verbose=2)


Y_hat = rnn_model.predict(X_tr, verbose=2).reshape(-1)
train_acc = (np.round(Y_hat) == Y_tr).mean()
Y_pred = rnn_model.predict(X_te, verbose=2).reshape(-1)
(np.round(Y_pred) == Y_te).mean()
pred_acc = (np.round(Y_pred) == Y_te).mean()


# Collect all hidden layers ################################################################
if use_TT:
    # Reconstruct the fully connected input-to-hidden weights:
    from keras.initializers import constant
    _tt_output_shape = np.copy(tt_output_shape)
    _tt_output_shape[0] *= 4

    fc_w = rnn_model.get_weights()[0]
    fc_layer = TT_Layer(tt_input_shape=tt_input_shape, tt_output_shape=_tt_output_shape, tt_ranks=tt_ranks,
                            kernel_initializer=constant(value=fc_w), use_bias=False)
    fc_input = Input(shape=(X_tr.shape[2],))
    fc_output = fc_layer(fc_input)
    fc_model = Model(fc_input, fc_output)
    fc_model.compile('sgd', 'mse')

    fc_recon_mat = fc_model.predict(np.identity(X_tr.shape[2]))

    # Reconstruct the entire LSTM:
    fc_Z = LSTM(units=np.prod(tt_output_shape), return_sequences=False, dropout=.5, recurrent_dropout=.5,
                weights=[fc_recon_mat, rnn_model.get_weights()[2], rnn_model.get_weights()[1]])(X_mask)

else:
    fc_Z = LSTM(units=rnn_size, return_sequences=False, dropout=.5, recurrent_dropout=.5,
                weights=rnn_model.get_weights()[0:3])(X_mask)

fc_Out = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(1e-3),
               weights=rnn_model.get_weights()[3::])(fc_Z)
fc_rnn_model = Model(X, fc_Out)
fc_rnn_model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy',
                  metrics=['accuracy'])

fc_rnn_model.evaluate(X_te, Y_te, verbose=2)



# Calculate the LRP: #########################################################################
fc_Z_model = Model(X, fc_Z)
fc_Z_model.compile('sgd', 'mse')

Y_hat_fc = fc_rnn_model.predict(X_tr)
Y_pred_fc = fc_rnn_model.predict(X_te)

Ws = fc_rnn_model.get_weights()[0]
Us = fc_rnn_model.get_weights()[1]
b = fc_rnn_model.get_weights()[2]
Dense_w = fc_rnn_model.get_weights()[3]
Dense_b = fc_rnn_model.get_weights()[4]

Z_tr = fc_Z_model.predict(X_tr)
Z_te = fc_Z_model.predict(X_te)

eps = 1e-4

is_number_flag = np.where(d_te != -1)

# All relevance scores of the test sequences
lrp_te = np.vstack([lstm_lrp(i, rnn_size, False).sum(1) for i in range(X_te.shape[0])])

lrp_auroc = roc_auc_score((d_te == 0).astype('int')[is_number_flag].reshape(-1),
                           lrp_te[is_number_flag].reshape(-1))
lrp_auprc = average_precision_score((d_te == 0).astype('int')[is_number_flag].reshape(-1),
                           lrp_te[is_number_flag].reshape(-1))


# The reported results:
print pred_acc
print lrp_auroc
print lrp_auprc

