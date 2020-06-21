from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from math import pi
from tensorflow.keras import layers
try:
	from tensorflow.keras import initializations
except ImportError:
	from tensorflow.keras import initializers as initializations
import tensorflow.keras.backend as K

class LocalPlanarGuidance(layers.Conv2D):
	def __init__(self, height, width, upratio, **kwargs):
		super(LocalPlanarGuidance, self).__init__(filters=3, kernel_size=1, dilation_rate=1, padding='same', activation='sigmoid', **kwargs)
		self.full_height = height
		self.full_width = width
		self.upratio = upratio

	def build(self, input_shape):
		self.batch_size = input_shape[0]
		v, u = tf.meshgrid(
			np.linspace(0, self.full_width -1, self.full_width, dtype=np.float32), 
			np.linspace(0, self.full_height -1, self.full_height, dtype=np.float32))

		v = K.expand_dims(K.stack([v]*self.batch_size, axis=0), axis=-1)
		v = (v % self.upratio - (self.upratio - 1)/2) / K.reshape(float(self.upratio), (-1, 1, 1, 1)) # V2 Update

		u = K.expand_dims(K.stack([u]*self.batch_size, axis=0), axis=-1)
		u = (u % self.upratio - (self.upratio - 1)/2) / K.reshape(float(self.upratio), (-1, 1, 1, 1)) # V2 Update

		self.pixel_dir_unit = K.l2_normalize(K.concatenate([u, v, K.ones_like(u)], axis=3), axis=3) # Normalize first

		return super(LocalPlanarGuidance, self).build(input_shape)

	def call(self, inputs):
		conv_sigmoid = super(LocalPlanarGuidance, self).call(inputs)
		# Decrease max value of theta to pi/3 to enhance numerical stability
		phi, theta, raw_dist = conv_sigmoid[..., 0] * 2 * pi, conv_sigmoid[..., 1] * pi / 3, conv_sigmoid[..., 2]
		plane_coeffs = K.stack([K.sin(theta) * K.cos(phi), K.sin(theta) * K.sin(phi), K.cos(theta), raw_dist], axis=-1)
		
		plane_exp_height = K.repeat_elements(plane_coeffs, self.upratio, axis=1)
		plane_exp = K.repeat_elements(plane_exp_height, self.upratio, axis=2)

		denominator = K.sum(self.pixel_dir_unit*plane_exp[..., 0:3], axis=-1, keepdims=True) + K.epsilon()
		return plane_exp[..., 3:] / denominator

	def get_config(self):
		base_config = super(LocalPlanarGuidance, self).get_config()
		config = {'upratio': self.upratio, 'full_height': self.full_height, 'full_width': self.full_width}
		return dict(list(base_config.items()) + list(config.items()))

class Scale(layers.Layer):
	'''Custom Layer for DenseNet used for BatchNormalization.
	
	Learns a set of weights and biases used for scaling the input data.
	the output consists simply in an element-wise multiplication of the input
	and a sum of a set of constants:

		out = in * gamma + beta,

	where 'gamma' and 'beta' are the weights and biases larned.

	# Arguments
		axis: integer, axis along which to normalize in mode 0. For instance,
			if your input tensor has shape (samples, channels, rows, cols),
			set axis to 1 to normalize per feature map (channels axis).
		momentum: momentum in the computation of the
			exponential average of the mean and standard deviation
			of the data, for feature-wise normalization.
		weights: Initialization weights.
			List of 2 Numpy arrays, with shapes:
			`[(input_shape,), (input_shape,)]`
		beta_init: name of initialization function for shift parameter
			(see [initializations](../initializations.md)), or alternatively,
			Theano/TensorFlow function to use for weights initialization.
			This parameter is only relevant if you don't pass a `weights` argument.
		gamma_init: name of initialization function for scale parameter (see
			[initializations](../initializations.md)), or alternatively,
			Theano/TensorFlow function to use for weights initialization.
			This parameter is only relevant if you don't pass a `weights` argument.
	'''
	def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
		self.momentum = momentum
		self.axis = axis
		self.beta_init = initializations.get(beta_init)
		self.gamma_init = initializations.get(gamma_init)
		self.initial_weights = weights
		super(Scale, self).__init__(**kwargs)

	def build(self, input_shape):
		shape = (int(input_shape[self.axis]),)

		self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
		self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def call(self, x, mask=None):
		broadcast_shape = [1] * len(x.shape)
		broadcast_shape[self.axis] = x.shape[self.axis]

		out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
		return out

	def get_config(self):
		config = {"momentum": self.momentum, "axis": self.axis}
		base_config = super(Scale, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
