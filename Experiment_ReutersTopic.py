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
An MLP where the first layer can be replaced by TT_Layer.
With current hyper-parameter settings we have:

With TT_Layer:
Runtime per epoch: 2-3 seconds
Runtime in total (with potential early stopping): 0:02:33
Number of parameters: 9600
AUROC: 0.956452509027
AUPRC: 0.510159794093
Accuracy: 0.776936776492

Without TT_Layer:
Runtime per epoch: 20-23 seconds
Runtime in total (with potential early stopping): 0:12:11
Number of parameters: 512000
AUROC: 0.954760401576
AUPRC: 0.548228866191
Accuracy: 0.788067675868

Compression factor: 0.01875
"""



import numpy as np
np.random.seed(11111986)
import datetime

# TT Layers
from TTLayer import TT_Layer

# Keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

max_words = 1000
batch_size = 16

(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)

nb_classes = np.max(y_train)+1

tokenizer = Tokenizer(nb_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

col_shuffle = np.random.choice(range(X_train.shape[1]), X_train.shape[1], False)
X_train = X_train[:, col_shuffle]
X_test = X_test[:, col_shuffle]



use_TT = False

input = Input(shape=(max_words, ))
if use_TT:
    h = TT_Layer(tt_input_shape=[10, 10, 10], tt_output_shape=[8, 8, 8], tt_ranks=[1, 10, 10, 1],
                 ortho_init=True, activation='relu')(input)
else:
    h = Dense(output_dim=512, activation='relu')(input)
h = BatchNormalization()(h)
output = Dense(output_dim=nb_classes, activation='softmax', kernel_regularizer=l2(1e-1))(h)
model = Model(input, output)
model.compile(optimizer=Adam(3e-4), loss='categorical_crossentropy', metrics=['accuracy'])

maxIter = 50

loss = np.zeros((maxIter,))
val_loss = np.zeros((maxIter,))
earlystop_thresh = .01
earlystop_steps = 5
best_val = np.Inf
earlystop_prop = 0.05
earlystop_count = 0


start = datetime.datetime.now()
for l in range(maxIter):
    print 'iter ' + str(l)
    history = model.fit(X_train, Y_train, epochs=1, batch_size=batch_size,
                        verbose=2, validation_split=0.2)

    loss[l] = history.history['loss'][0]
    val_loss[l] = history.history['val_loss'][0]

    if val_loss[l] < best_val:
        best_val = val_loss[l]
        earlystop_count = 0
    else:
        if ( val_loss[l] / best_val -1 > earlystop_prop):
            earlystop_count += 1
    if earlystop_count == earlystop_steps:
        print 'Early stopped after ' + str(l) + ' iterations!'
        break

    if l>0 and l % 10 == 0:
        score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=2)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

stop = datetime.datetime.now()

# score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=2)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

print stop - start
Y_pred = model.predict(X_test)
print roc_auc_score(Y_test, Y_pred)
print average_precision_score(Y_test, Y_pred)
print accuracy_score(Y_test.argmax(1), Y_pred.argmax(1))
