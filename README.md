# TT_Layer
An implementation of the Tensor-Train Layer and Tensor-Train Recurrent Neural Networks for Keras, based on the existing implementation for Lasagne and Matlab.

## Reference:
        "Tensorizing Neural Networks"
        Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry Vetrov
        In Advances in Neural Information Processing Systems 28 (NIPS-2015).

        "Tensor-Train Recurrent Neural Networks for Video Classification"
        Yinchong Yang, Denis Krompass, Volker Tresp
        In International Conference on Machine Learning 34 (ICML-2017).

## Introduction:
A Tensor-Train Layer (TTL) provides a replacement of a fully-connected feed-forward layer in Neural Networks. It factorizes the otherwise dense weight matrix into a train of smaller, 4D tensors and can therefore reduce the number of weights in the model. It is expected to speed up the training and inference to a rather large and tunable proportion, while not sacrificing much of the modeling power.
This is especially the case, when the input contains redundant information, which could be e.g. pixels in a image or feature map after convolution. Also for feature selection / dimension reduction tasks that are too expensive to run lasso- and ridge-regularization, TT layer could also be an option so long as one is more interested in the mere prediction quality than identifying specific explaining features.

A Tensor-Train Recurrent Neural Network (TT-RNN), including LSTM and GRU, replaces the weight matrix mapping from input to hidden layer (the gates in case of LSTM and GRU) with a TTL. This turns out to enable RNNs to handle very high-dimensional sequential data, such as video streams, where each frame in a video forms an extremely high-dimensional input vector. 

## Implementation:
Our TTLayer.py is a reimplementation of the terrific work by: https://github.com/Bihaqo/TensorNet with specific improvement and adjustments for Keras. Applying only keras grammar i.e. the Backend functions, this implementation is expected to utilize both Theano and TensorFlow(note yet tested) backends and compatible with other keras components. 
We further integrate this code into the keras RNN implementations: TTRNN.py.  

## Usage:
Once you've decided the Tensor-Train setting, including the factorization of the input and output vector as well as the tensor ranks, you can define the TT layer just in place of a usual dense one:
```python
h = TT_Layer(tt_input_shape=[4, 8, 8, 4], tt_output_shape=[4, 8, 8, 4], tt_ranks=[1, 3, 3, 3, 1],
              use_bias=True, activation='relu', W_regularizer=l2(5e-4), debug=False, ortho_init=True)
```
and use the TTRNNs as if they were usual RNNs: 
```python
rnn_layer = TT_GRU(tt_input_shape=[16, 32], tt_output_shape=[8, 8], tt_ranks=[1, 3, 1],
                   return_sequences=False, debug=True,
                   dropout=.25, recurrent_dropout=.25, activation='tanh')
```

## Experiments:
### For TTLayer we provide following experiments: 
### 1. [NIPS2003 Feature Selection Challenge](../master/Experiment_NIPS2003.py)
  including 4 out of 5 Datasets at the NIPS2003 workshop on feature selection
### 2. [Gene Expression Data](../master/Experiment_GenExpress.py)
  including 4 Datasets that are from the R-package spls
### 3. [MNIST Data](../master/Experiment_MNIST.py)
### 4. [CIFAR10 Data](../master/Experiment_CIFAR10.py)
### 5. [Reuters News Topic Data](../master/Experiment_ReutersTopic.py)

### For TT-RNNs we provide following experiments as in our latest paper: 
### 1. [UCF11](../master/Experiment_UCF11.py)
### 2. [Hollywood2](../master/Experiment_Hollywood2.py)
### 3. [Youtube Celebrity Faces](../master/Experiment_CelebFaces.py)
### 4. [HMDB51](../master/Experiment_HMDB51.py)

## Acknoledgement:
I would like to thank Alexander Novikov and his colleagues for sharing their codes in lasagne and as well as a lot of first-hand experience in applying the model, without which this implementation would not have been possible. Also many thanks go to Siemens AG for supporting this project.

## Notes
This implementation is still at a very early stage. There might be some warnings regarding the keras 2.x, because the code was originally developed in keras 1.x. We therefore highly appreciate insights, recommendations and critics of all kinds.

## Dependency Versions: 
Tested with Keras 1.0.8 and Theano 0.8.2.dev., partly Keras 2.0.4 and Theano 0.9.0..
