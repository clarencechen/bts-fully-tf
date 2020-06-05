#! -*- coding: utf-8 -*-

import os, sys
#from distutils.util import strtobool
import numpy as np
import tensorflow as tf

# Using tf.keras in this case
is_tf_keras = True

if is_tf_keras:
	import tensorflow.keras as keras
	import tensorflow.keras.backend as K
	sys.modules['keras'] = keras
else:
	import keras
	import keras.backend as K


class AdaFactorBase(keras.optimizers.Optimizer):
	"""AdaFactor optmizer（Base Class）
	Link to paper：https://arxiv.org/abs/1804.04235
	Implementation (annotated in Chinese)：https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
	"""
	def __init__(
			self,
			lr=1e-3,  # may be None
			beta1=0.0,
			beta2=None,
			epsilon1=1e-30,
			epsilon2=1e-3,
			multiply_by_parameter_scale=True,
			clipping_threshold=1.0,
			min_dim_size_to_factor=128,
			**kwargs):
		super(AdaFactorBase, self).__init__(**kwargs)
		self._lr = lr
		if self._lr is not None:
			self._lr_tensor = K.variable(self._lr, name='lr')
		self.beta1 = beta1
		self._beta2 = beta2
		self.epsilon1 = epsilon1
		self.epsilon2 = epsilon2
		self.multiply_by_parameter_scale = multiply_by_parameter_scale
		self.clipping_threshold = clipping_threshold
		self.min_dim_size_to_factor = min_dim_size_to_factor

	@property
	def lr(self):
		if self._lr is None:
			iterations = K.cast(self.iterations + 1, K.floatx())
			lr_value = K.minimum(1.0 / K.sqrt(iterations), 0.01)
			if self.multiply_by_parameter_scale:
				self._lr_tensor = lr_value
			else:
				self._lr_tensor = lr_value * 0.05
		return self._lr_tensor

	@property
	def beta2(self):
		if self._beta2 is None:
			iterations = K.cast(self.iterations + 1, K.floatx())
			return 1.0 - K.pow(iterations, -0.8)
		else:
			return self._beta2

	def factored_shape(self, shape):
		if len(shape) < 2:
			return None
		shape = np.array(shape)
		indices = shape.argpartition(-2)
		if indices[-2] < self.min_dim_size_to_factor:
			return None
		shape1, shape2 = np.array(shape), np.array(shape)
		shape1[indices[-1]] = 1
		shape2[indices[-2]] = 1
		return shape1, indices[-1], shape2, indices[-2]

	def get_config(self):
		config = {
			'lr': self._lr,
			'beta1': self.beta1,
			'beta2': self._beta2,
			'epsilon1': self.epsilon1,
			'epsilon2': self.epsilon2,
			'multiply_by_parameter_scale': self.multiply_by_parameter_scale,
			'clipping_threshold': self.clipping_threshold,
			'min_dim_size_to_factor': self.min_dim_size_to_factor,
		}
		base_config = super(AdaFactorBase, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class AdaFactorV2(AdaFactorBase):
	"""AdaFactor optmizer（tf.keras version）
	Link to paper：https://arxiv.org/abs/1804.04235
	Implementation (annotated in Chinese)：https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py
	"""
	def __init__(self, *args, **kwargs):
		if 'name' not in kwargs:
			kwargs['name'] = 'AdaFactor'
		super(AdaFactorV2, self).__init__(*args, **kwargs)

	def _create_slots(self, var_list):
		for var in var_list:
			if self.beta1 > 0.0:
				self.add_slot(var, 'm')
			shape = K.int_shape(var)
			factored_shape = self.factored_shape(shape)
			if factored_shape is None:
				self.add_slot(var, 'v')
			else:
				shape1, axis1, shape2, axis2 = factored_shape
				value1, value2 = np.zeros(shape1), np.zeros(shape2)
				self.add_slot(var, 'vr', value1)
				self.add_slot(var, 'vc', value2)

	def _resource_apply_dense(self, grad, var):
		g2 = K.square(grad) + self.epsilon1
		shape = K.int_shape(var)
		factored_shape = self.factored_shape(shape)
		if factored_shape is None:
			v = self.get_slot(var, 'v')
			# Define aux variable
			v_t = self.beta2 * v + (1.0 - self.beta2) * g2
			v_t = K.update(v, v_t)
		else:
			shape1, axis1, shape2, axis2 = factored_shape
			vr = self.get_slot(var, 'vr')
			vc = self.get_slot(var, 'vc')
			# Define aux variable
			vr_t = self.beta2 * vr + K.mean(g2, axis=axis1, keepdims=True)
			vc_t = self.beta2 * vc + K.mean(g2, axis=axis2, keepdims=True)
			vr_t, vc_t = K.update(vr, vr_t), K.update(vc, vc_t)
			# Compose full v matrix
			v_t = vr_t * vc_t / K.mean(vr_t, axis=axis2, keepdims=True)
		# Define main update
		u = grad / K.sqrt(v_t)
		# Define clipping
		if self.clipping_threshold is not None:
			u_rms = K.mean(K.sum(K.square(u)))
			d = self.clipping_threshold
			u = u / K.maximum(1.0, u_rms / d)
		# Define momentum
		if self.beta1 > 0.0:
			m = self.get_slot(var, 'm')
			# Define aux variable
			m_t = self.beta1 * m + (1.0 - self.beta1) * u
			u = K.update(m, m_t)
		# Define parameter scaling
		if self.multiply_by_parameter_scale:
			u = u * K.maximum(K.mean(K.sum(K.square(var))), self.epsilon2)
		# Update parameters
		return K.update(var, var - self.lr * u)

	def _resource_apply_sparse(self, grad, var, indices):
		grad = tf.IndexedSlices(grad, indices, K.shape(var))
		grad = tf.convert_to_tensor(grad)
		return self._resource_apply_dense(grad, var)

AdaFactor = AdaFactorV2