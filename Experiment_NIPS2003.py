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
A comparison between TT layer and dense layer is conducted using the datasets from
    NIPS 2003 workshop on feature extraction (http://clopinet.com/isabelle/Projects/NIPS2003/)
There are in total 5 datasets involved: arcene, dexter, dorothea, gisette and madelon. For more details
please kindly refer to the website above or the corresponding directory at UCI Machine learning repositroy.

With each dataset two 2-layered NN models are trained. The first one is a standard NN with two dense
layers while the second model replaces the first dense layer with a TT layer. Both models share the
same, however rather naiive hyper parameter settings without further fine tuning. The purpose is to
show that for features that may contain redundant and/or irrelevant information, or rather, with
specific features being more relevant than others, replacing dense layers with TT layers can speed
up the training to a large and tunable proportion, without sacrificing much of the modeling quality.

In term of the runtime we do not, however, consider the fact that the models may have been using
multiple cores(up to 24). Since the TT layers apparently tend to take less parameters, its advantage
will be even more obvious when the model is forced to be trained on a single core. This aspect will
be included in future development.
The parameter compression factor is calculated:
number of parameter in the dense layer / number of parameters in the TT layer

TT layer may provide an alternative to usual regularization methods, as long as one is more interested
in the modeling quality than identifying the important features. This could especially be the case
when the input data are of very high dimensionality, making training the model with regularization
very time-intensive in the first place.

Experimental results:
(We use only report those of the first 4 data sets, since we are still looking for the appropriate
hyper parameter setting. Therefore any )

arcene -------------------------------------------------------------------------------------------

Results of the model with fully connected layer
Time consumed: 0:06:24.052958
Accuracy: 0.84
AUROC: 0.932224025974
AUPRC: 0.93266485352

Results of the model with TT layer
Time consumed: 0:00:52.214801
Accuracy: 0.82
AUROC: 0.910308441558
AUPRC: 0.900333978362

Parameter compression factor: 11000 / 6250000 = 0.00176

gisette -------------------------------------------------------------------------------------------

Results of the model with fully connected layer
Time consumed: 0:03:34.768943
Accuracy: 0.902
AUROC: 0.965552
AUPRC: 0.966865457207

Results of the model with TT layer
Time consumed: 0:02:36.213685
Accuracy: 0.927
AUROC: 0.979928
AUPRC: 0.978744908369

Parameter compression factor: 1725 / 405000 = 0.00425925925926


dexter -------------------------------------------------------------------------------------------
Results of the model with fully connected layer
Time consumed: 0:08:30.379006
Accuracy: 0.6
AUROC: 0.971555555556
AUPRC: 0.970893370868

Results of the model with TT layer
Time consumed: 0:07:55.489709
Accuracy: 0.73
AUROC: 0.810488888889
AUPRC: 0.782389136715

Parameter compression factor: 9270 / 4860000 = 0.00190740740741

dorothea ------------------------------------------------------------------------------------------
Results of the model with fully connected layer
Time consumed: 0:20:00.062163
Accuracy: 0.908571428571
AUROC: 0.932055100521
AUPRC: 0.723815331211

Results of the model with TT layer
Time consumed: 0:08:03.457054
Accuracy: 0.902857142857
AUROC: 0.731291883842
AUPRC: 0.325179205933

Parameter compression factor: 23250 / 62500000 = 0.000372

# Note this time the AUPRC is more trustworthy because the target is unbalanced. To this extent the
FC model is better.

# madelon -----------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------

There seems to be large margin of improvement in term of finer tuning the hyper parameters.
Insights, recommendations and critics are welcome and highly appreciated.

"""

# Basic
import numpy as np
from datetime import datetime

# Keras Model
from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

# TT Layer
from TTLayer import TT_Layer

# Data
from Datasets.Datasets import *

# misc
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

# run_local = int(sys.argv[1]) # 1 for local; 0 for server
# data_name = sys.argv[2] # arcene or gisette

np.random.seed(11111986)

run_local = 0

# Choose one data set:
data_name = 'arcene'
# data_name = 'gisette'
# data_name = 'dexter'
# data_name = 'dorothea'
# data_name = 'madelon'


if run_local == 0:  # if not local i.e. on a server without internet, read everything from .gz files
    data_path = './Datasets/NIPS2003/'
    if data_name in ['arcene', 'gisette', 'madelon']:
        X_train = np.loadtxt(data_path + data_name + '/' + data_name + '_train.data.gz')
        Y_train = np.loadtxt(data_path + data_name + '/' + data_name + '_train.labels.gz')
        X_valid = np.loadtxt(data_path + data_name + '/' + data_name + '_valid.data.gz')
        Y_valid = np.loadtxt(data_path + data_name + '/' + data_name + '_valid.labels.gz')
    elif data_name == 'dexter':
        X_train = get_dexter_data(data_path + 'dexter/dexter_train.data.gz', mode='gz')
        Y_train = np.loadtxt(data_path + 'dexter/dexter_train.labels.gz')
        X_valid = get_dexter_data(data_path + 'dexter/dexter_valid.data.gz', mode='gz')
        Y_valid = np.loadtxt(data_path + 'dexter/dexter_valid.labels.gz')
    elif data_name == 'dorothea':
        X_train = get_dorothea_data(data_path + 'dorothea/dorothea_train.data.gz', mode='gz')
        Y_train = np.loadtxt(data_path + 'dorothea/dorothea_train.labels.gz')
        X_valid = get_dorothea_data(data_path + 'dorothea/dorothea_valid.data.gz', mode='gz')
        Y_valid = np.loadtxt(data_path + 'dorothea/dorothea_valid.labels.gz')
else:  # otherwise download the files from repo
    X_train, Y_train, X_valid, Y_valid = load_NIPS2003_data(data_name)

n, d = X_train.shape
print 'Training data has shape = ' + str(X_train.shape)
print 'Valid data has shape = ' + str(X_valid.shape)

# Two possibilities to normalize the X for datasets other than dorothea:
# either 0) into the range [0,1] or 1) using a standard gaussian
normalization = 1

if data_name not in ['dorothea']:  # dorothea is binary, no need for normalization
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

# replace the -1 in the original labels with 0
Y_train[np.where(Y_train == -1.)[0]] = 0.
Y_train = Y_train.astype('int32')
Y_valid[np.where(Y_valid == -1.)[0]] = 0.
Y_valid = Y_valid.astype('int32')


# Hyper parameter settings for each data set
if data_name == 'arcene':
    alpha = 0.01
    tt_alpha = 5e-4
    nb_epoch = 200
    batch_size = 5
    lr = 1e-4
    h_dropout = 0
    tt_input_shape = [10, 10, 10, 10]
    tt_output_shape = [5, 5, 5, 5]
    tt_ranks = [1, 10, 10, 10, 1]
elif data_name == 'gisette':
    alpha = 0.1
    tt_alpha = 5e-4
    nb_epoch = 200
    batch_size = 25
    lr = 1e-4
    h_dropout = 0
    tt_input_shape = [5, 10, 10, 10]
    tt_output_shape = [3, 3, 3, 3]
    tt_ranks = [1, 5, 5, 5, 1]
elif data_name == 'dexter':
    alpha = 0.1
    tt_alpha = 5e-4
    nb_epoch = 400
    batch_size = 35
    lr = 1e-4
    h_dropout = 0
    tt_input_shape = [4, 10, 10, 10, 5]
    tt_output_shape = [3, 3, 3, 3, 3]
    tt_ranks = [1, 10, 10, 10, 10, 1]
elif data_name == 'dorothea':
    alpha = 0.01
    tt_alpha = 5e-4
    nb_epoch = 100
    batch_size = 45
    lr = 1e-4
    h_dropout = 0
    tt_input_shape = [10, 20, 50, 10]
    tt_output_shape = [5, 5, 5, 5]
    tt_ranks = [1, 10, 10, 5, 1]
elif data_name == 'madelon':
    alpha = 0.005
    tt_alpha = 5e-3
    nb_epoch = 600
    batch_size = 20
    lr = 1e-4
    h_dropout = 0
    tt_input_shape = [5, 5, 5, 4]
    tt_output_shape = [3, 3, 3, 3]
    tt_ranks = [1, 5, 5, 5, 1]

# Model with fully connected layer

train_loss_full = np.zeros(nb_epoch)
valid_loss_full = np.zeros(nb_epoch)
test_loss_full = np.zeros(nb_epoch)

train_acc_full = np.zeros(nb_epoch)
valid_acc_full = np.zeros(nb_epoch)
test_acc_full = np.zeros(nb_epoch)

np.random.seed(11111986)
input = Input(shape=(d,))
h = Dense(output_dim=np.prod(tt_output_shape), activation='sigmoid', kernel_regularizer=l2(alpha))(input)
# h = Dropout(h_dropout)(h)
output = Dense(output_dim=1, activation='sigmoid', kernel_regularizer=l2(alpha))(h)
model_full = Model(input=input, output=output)
model_full.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

start_full = datetime.now()
for l in range(nb_epoch):

    if_print = l % 10 == 0
    if if_print:
        print 'iter = ' + str(l)
        verbose = 2
    else:
        verbose = 0
    history = model_full.fit(x=X_train, y=Y_train, verbose=verbose, nb_epoch=1, batch_size=batch_size,
                             validation_split=0.2)
    train_loss_full[l] = history.history['loss'][0]
    valid_loss_full[l] = history.history['val_loss'][0]
    train_acc_full[l] = history.history['acc'][0]
    valid_acc_full[l] = history.history['val_acc'][0]

    eval_full = model_full.evaluate(X_valid, Y_valid, batch_size=X_valid.shape[0], verbose=2)
    test_loss_full[l] = eval_full[0]
    test_acc_full[l] = eval_full[1]

stop_full = datetime.now()


# Model with TT layer

train_loss_TT = np.zeros(nb_epoch)
valid_loss_TT = np.zeros(nb_epoch)
test_loss_TT = np.zeros(nb_epoch)

train_acc_TT = np.zeros(nb_epoch)
valid_acc_TT = np.zeros(nb_epoch)
test_acc_TT = np.zeros(nb_epoch)

np.random.seed(11111986)
input_TT = Input(shape=(d,))
tt = TT_Layer(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape, kernel_regularizer=l2(tt_alpha),
              tt_ranks=tt_ranks, bias=True, activation='sigmoid', ortho_init=True)
h_TT = tt(input_TT)
# h_TT = Dropout(h_dropout)(h_TT)
output_TT = Dense(output_dim=1, activation='sigmoid', kernel_regularizer=l2(alpha))(h_TT)
model_TT = Model(input=input_TT, output=output_TT)
model_TT.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

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
    train_loss_TT[l] = history.history['loss'][0]
    valid_loss_TT[l] = history.history['val_loss'][0]
    train_acc_TT[l] = history.history['acc'][0]
    valid_acc_TT[l] = history.history['val_acc'][0]

    eval_TT = model_TT.evaluate(X_valid, Y_valid, batch_size=X_valid.shape[0], verbose=2)
    test_loss_TT[l] = eval_TT[0]
    test_acc_TT[l] = eval_TT[1]

stop_TT = datetime.now()


# print '#######################################################'
Y_pred_full = model_full.predict(X_valid)
print 'Results of the model with fully connected layer'
print 'Time consumed: ' + str(stop_full - start_full)
print 'Accuracy: ' + str(accuracy_score(Y_valid, np.round(Y_pred_full)))
print 'AUROC: ' + str(roc_auc_score(Y_valid, Y_pred_full))
print 'AUPRC: ' + str(average_precision_score(Y_valid, Y_pred_full))


# print '#######################################################'
Y_pred_TT = model_TT.predict(X_valid)
print 'Results of the model with TT layer'
print 'Time consumed: ' + str(stop_TT - start_TT)
print 'Accuracy: ' + str(accuracy_score(Y_valid, np.round(Y_pred_TT)))
print 'AUROC: ' + str(roc_auc_score(Y_valid, Y_pred_TT))
print 'AUPRC: ' + str(average_precision_score(Y_valid, Y_pred_TT))
print '\n'
print 'Parameter compression factor: ' + str(tt.TT_size) + '/' +  \
      str(tt.full_size) + ' = ' + str(tt.compress_factor)


