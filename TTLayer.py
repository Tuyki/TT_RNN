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


import numpy as np

import keras.activations
from keras import backend as K
from keras.engine.topology import Layer

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

def init_orthogonal_tt_cores(tt_input_shape, tt_output_shape, tt_ranks):
    tt_input_shape = np.array(tt_input_shape)
    tt_output_shape = np.array(tt_output_shape)
    tt_ranks = np.array(tt_ranks)
    cores_arr_len = np.sum(tt_input_shape * tt_output_shape *
                           tt_ranks[1:] * tt_ranks[:-1])
    cores_arr = np.zeros(cores_arr_len)
    rv = 1

    d = tt_input_shape.shape[0]
    rng = np.random
    shapes = [None] * d
    tall_shapes = [None] * d
    cores = [None] * d
    counter = 0

    for k in range(tt_input_shape.shape[0]):
        # Original implementation
        # shape = [ranks[k], input_shape[k], output_shape[k], ranks[k+1]]
        shapes[k] = [tt_ranks[k], tt_input_shape[k], tt_output_shape[k], tt_ranks[k + 1]]

        # Original implementation
        # tall_shape = (np.prod(shape[:3]), shape[3])
        tall_shapes[k] = (np.prod(shapes[k][:3]), shapes[k][3])

        # Original implementation
        # curr_core = np.dot(rv, np.random.randn(shape[0], np.prod(shape[1:])) )
        cores[k] = np.dot(rv, rng.randn(shapes[k][0], np.prod(shapes[k][1:])))

        # Original implementation
        # curr_core = curr_core.reshape(tall_shape)
        cores[k] = cores[k].reshape(tall_shapes[k])

        if k < tt_input_shape.shape[0] - 1:
            # Original implementation
            # curr_core, rv = np.linalg.qr(curr_core)
            cores[k], rv = np.linalg.qr(cores[k])
        # Original implementation
        # cores_arr[cores_arr_idx:cores_arr_idx+curr_core.size] = curr_core.flatten()
        # cores_arr_idx += curr_core.size
        cores_arr[counter:(counter + cores[k].size)] = cores[k].flatten()
        counter += cores[k].size

    glarot_style = (np.prod(tt_input_shape) * np.prod(tt_ranks)) ** (1.0 / tt_input_shape.shape[0])
    return (0.1 / glarot_style) * cores_arr


class TT_Layer(Layer):
    """
    # References:
    "Tensorizing Neural Networks"
    Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry Vetrov
    In Advances in Neural Information Processing Systems 28 (NIPS-2015).
    This a reimplementation of the work by:
    https://github.com/Bihaqo/TensorNet/blob/master/src/python/ttlayer.py
    with specific improvement and adjustments for Keras, which is rather promising framework for
    Deep Learning. Applying only keras grammar, this implementation is expected to utilize
    both Theano and TensorFlow(note yet tested) backends and compatible with other keras components.

    # Introduction:
    A Tensor-Train layer provides a replacement of a fully-connected (dense) layer. It factorizes
    the latter into a train of smaller, 4D tensors and can therefore reduce the total number of
    parameters in the original layer. It is expected to speed up the training and inference to a
    rather large and tunable proportion, while not sacrificing much modeling quality.
    This is especially the case, when the input contains redundant information, which could be e.g.
    pixels in a image or feature map after convolution.
    Also for feature selection / dimension reduction tasks that are too expensive to run lasso-
    and ridge-regularization, TT layer could also be an option so long as one is more interested in
    the mere prediction quality than identifying specific explaining features.

    # Arguments:
        tt_input_shape: a list of shapes, the product of which should be equal to the input dimension
        tt_output_shape: a list of shapes of the same length as tt_input_shape,
            the product of which should be equal to the output dimension
        tt_ranks: a list of length len(tt_input_shape)+1, the first and last rank should only be 1
        the rest of the arguments: please refer to dense layer in keras. 
    """

    def __init__(self, tt_input_shape, tt_output_shape, tt_ranks,
                 use_bias=True,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 debug=False,
                 init_seed=11111986,
                 **kwargs):

        tt_input_shape = np.array(tt_input_shape)
        tt_output_shape = np.array(tt_output_shape)
        tt_ranks = np.array(tt_ranks)

        self.tt_input_shape = tt_input_shape
        self.tt_output_shape = tt_output_shape
        self.tt_ranks = tt_ranks
        self.num_dim = tt_input_shape.shape[0]  # length of the train
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.debug = debug
        self.init_seed = init_seed

        super(TT_Layer, self).__init__(**kwargs)

    def build(self, input_shape):

        num_inputs = int(np.prod(input_shape[1::]))

        # Check the dimensionality
        if np.prod(self.tt_input_shape) != num_inputs:
            raise ValueError("The size of the input tensor (i.e. product "
                             "of the elements in tt_input_shape) should "
                             "equal to the number of input neurons %d." % num_inputs)
        if self.tt_input_shape.shape[0] != self.tt_output_shape.shape[0]:
            raise ValueError("The number of input and output dimensions "
                             "should be the same.")
        if self.tt_ranks.shape[0] != self.tt_output_shape.shape[0] + 1:
            raise ValueError("The number of the TT-ranks should be "
                             "1 + the number of the dimensions.")
        if self.debug:
            print 'tt_input_shape = ' + str(self.tt_input_shape)
            print 'tt_output_shape = ' + str(self.tt_output_shape)
            print 'tt_ranks = ' + str(self.tt_ranks)

        # Initialize the weights
        if self.init_seed is None:
            self.init_seed = 11111986
        np.random.seed(self.init_seed)
        # if self.ortho_init:
        #     local_cores_arr = self._generate_orthogonal_tt_cores()
        # else:
        #     total_length = np.sum(self.tt_input_shape * self.tt_output_shape *
        #                           self.tt_ranks[1:] * self.tt_ranks[:-1])
        #     local_cores_arr = np.random.randn(total_length, 1) * .01

        total_length = np.sum(self.tt_input_shape * self.tt_output_shape *
                                  self.tt_ranks[1:] * self.tt_ranks[:-1])
        # if self.ortho_init:
        #     self.kernel = self.add_weight((total_length,),
        #                                   initializer=self._generate_orthogonal_tt_cores,
        #                                   name='kernel',
        #                                   regularizer=self.kernel_regularizer,
        #                                   constraint=self.kernel_constraint)
        # else:
        self.kernel = self.add_weight((total_length, ),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight((np.prod(self.tt_output_shape), ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        # Pre-calculate the indices, shapes and cores
        self.inds = np.zeros(self.num_dim).astype('int32')
        self.shapes = np.zeros((self.num_dim, 2)).astype('int32')
        self.cores = [None] * self.num_dim

        for k in range(self.num_dim - 1, -1, -1):
            # This is the shape of (m_k * r_{k+1}) * (r_k * n_k)
            self.shapes[k] = (self.tt_input_shape[k] * self.tt_ranks[k + 1],
                              self.tt_ranks[k] * self.tt_output_shape[k])
            # Note that self.cores store only the pointers to the parameter vector
            self.cores[k] = self.kernel[self.inds[k]:self.inds[k] + np.prod(self.shapes[k])]
            if 0 < k:  # < self.num_dim-1:
                self.inds[k - 1] = self.inds[k] + np.prod(self.shapes[k])
        if self.debug:
            print 'self.shapes = ' + str(self.shapes)

        # Calculate and print the compression factor
        self.TT_size = total_length
        self.full_size = (np.prod(self.tt_input_shape) * np.prod(self.tt_output_shape))
        self.compress_factor = 1. * self.TT_size / self.full_size
        print 'Compression factor = ' + str(self.TT_size) + ' / ' \
              + str(self.full_size) + ' = ' + str(self.compress_factor)

    def call(self, x, mask=None):
        # theano.scan doesn't seem to work when intermediate results' changes in shape -- Alexander
        res = x
        # core_arr_idx = 0 # oroginal implementation, removed
        for k in range(self.num_dim - 1, -1, -1):
            """
            These are the original codes by Alexander, which calculate the shapes ad-hoc.
            Feel free to switch these codes on if one is interested in comparing performances.
            At least I only observe very small differences.
            # res is of size o_k+1 x ... x o_d x batch_size x i_1 x ... x i_k-1 x i_k x r_k+1
            curr_shape = (self.tt_input_shape[k] * self.tt_ranks[k + 1], self.tt_ranks[k] * self.tt_output_shape[k])
            curr_core = self.W[core_arr_idx:core_arr_idx+K.prod(curr_shape)].reshape(curr_shape)
            res = K.dot(res.reshape((-1, curr_shape[0])), curr_core)
            # res is of size o_k+1 x ... x o_d x batch_size x i_1 x ... x i_k-1 x r_k x o_k
            res = K.transpose(res.reshape((-1, self.tt_output_shape[k])))
            # res is of size o_k x o_k+1 x ... x o_d x batch_size x i_1 x ... x i_k-1 x r_k
            core_arr_idx += K.prod(curr_shape)
            """
            # New one, in order to avoid calculating the indices in every iteration
            res = K.dot(K.reshape(res, (-1, self.shapes[k][0])),  # of shape (-1, m_k*r_{k+1})
                        K.reshape(self.cores[k], self.shapes[k])  # of shape (m_k*r_{k+1}, r_k*n_k)
                        )
            res = K.transpose(
                K.reshape(res, (-1, self.tt_output_shape[k]))
            )

        # res is of size o_1 x ... x o_d x batch_size # by Alexander
        res = K.transpose(K.reshape(res, (-1, K.shape(x)[0])))

        if self.use_bias:
            res = K.bias_add(res, self.bias)
        if self.activation is not None:
            res =self.activation(res)

        return res

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(self.tt_output_shape))

    def compute_output_shape(self, input_shape):
        # assert input_shape and len(input_shape) >= 2
        # assert input_shape[-1]
        # output_shape = list(input_shape)
        # output_shape[-1] = self.units
        return (input_shape[0], np.prod(self.tt_output_shape))

    def _generate_orthogonal_tt_cores(self):
        cores_arr_len = np.sum(self.tt_input_shape * self.tt_output_shape *
                               self.tt_ranks[1:] * self.tt_ranks[:-1])
        cores_arr = np.zeros(cores_arr_len)
        rv = 1

        d = self.tt_input_shape.shape[0]
        rng = np.random
        shapes = [None] * d
        tall_shapes = [None] * d
        cores = [None] * d
        counter = 0

        for k in range(self.tt_input_shape.shape[0]):
            # Original implementation
            # shape = [ranks[k], input_shape[k], output_shape[k], ranks[k+1]]
            shapes[k] = [self.tt_ranks[k], self.tt_input_shape[k], self.tt_output_shape[k], self.tt_ranks[k + 1]]

            # Original implementation
            # tall_shape = (np.prod(shape[:3]), shape[3])
            tall_shapes[k] = (np.prod(shapes[k][:3]), shapes[k][3])

            # Original implementation
            # curr_core = np.dot(rv, np.random.randn(shape[0], np.prod(shape[1:])) )
            cores[k] = np.dot(rv, rng.randn(shapes[k][0], np.prod(shapes[k][1:])))

            # Original implementation
            # curr_core = curr_core.reshape(tall_shape)
            cores[k] = cores[k].reshape(tall_shapes[k])

            if k < self.tt_input_shape.shape[0] - 1:
                # Original implementation
                # curr_core, rv = np.linalg.qr(curr_core)
                cores[k], rv = np.linalg.qr(cores[k])
            # Original implementation
            # cores_arr[cores_arr_idx:cores_arr_idx+curr_core.size] = curr_core.flatten()
            # cores_arr_idx += curr_core.size
            cores_arr[counter:(counter + cores[k].size)] = cores[k].flatten()
            counter += cores[k].size

        glarot_style = (np.prod(self.tt_input_shape) * np.prod(self.tt_ranks)) ** (1.0 / self.tt_input_shape.shape[0])
        return (0.1 / glarot_style) * cores_arr

    def get_full_W(self):
        res=np.identity(np.prod(self.tt_input_shape))
        for k in range(self.num_dim - 1, -1, -1):
            res = np.dot(np.reshape(res, (-1, self.shapes[k][0])),  # of shape (-1, m_k*r_{k+1})
                        np.reshape(self.cores[k], self.shapes[k])  # of shape (m_k*r_{k+1}, r_k*n_k)
                        )
            res = np.transpose(
                np.reshape(res, (-1, self.tt_output_shape[k]))
            )
        res = np.transpose(np.reshape(res, (-1, np.shape(res)[0])))

        if self.use_bias:
            res = K.bias_add(res, self.bias)
        if self.activation is not None:
            res =self.activation(res)

        return res


# Test
# X = np.random.randn(100, 256)
# B = np.random.randn(256, 16)
# Y = np.dot(X, B)
# from keras.models import Model
# from keras.layers import Input, Dense
# input = Input((256,))
# output = TT_Layer(tt_input_shape=[4, 4, 4, 4], tt_output_shape=[2, 2, 2, 2], tt_ranks=[1, 4, 4, 4, 1],
#                   use_bias=True, ortho_init=False, debug=True)(input)
# model = Model(input, output)
# model.compile('sgd', 'mse')
# model.fit(x=X, y=Y, batch_size=32, epochs=100)
