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


class AdamWBase(keras.optimizers.Optimizer):
	"""AdamW optmizer with beta2 schedule, parameter scaling, and update clipping (Base Class)
	Link to paper: https://arxiv.org/abs/1711.05101
	"""
	def __init__(
			self,
			lr=1e-3,  # may be None
			decay_var_list=None, # may be None
			beta1=0.9,
			beta2=None,
			epsilon=1e-7,
			epsilon_param_scale=1e-3,
			multiply_by_parameter_scale=True,
			clipping_threshold=1.0,
			**kwargs):
		super(AdamWBase, self).__init__(**kwargs)
		self._lr = lr
		if self._lr is not None:
			self._lr_tensor = K.variable(self._lr, name='lr')
		else:
			self._lr_tensor = K.variable(0.01, name='lr')
		self.decay_var_list = decay_var_list or {}
		self.beta1 = beta1
		self._beta2 = beta2
		self.epsilon = epsilon
		self.epsilon_param_scale = epsilon_param_scale
		self.multiply_by_parameter_scale = multiply_by_parameter_scale
		self.clipping_threshold = clipping_threshold

	@property
	def lr(self):
		if self._lr is None:
			iterations = K.cast(self.iterations + 1, K.floatx())
			lr_value = K.minimum(1.0 / K.sqrt(iterations), 0.01)
			if self.multiply_by_parameter_scale:
				K.update(self._lr_tensor, lr_value)
			else:
				K.update(self._lr_tensor, lr_value * 0.05)
		return self._lr_tensor

	@property
	def beta2(self):
		if self._beta2 is None:
			iterations = K.cast(self.iterations + 1, K.floatx())
			return 1.0 - K.pow(iterations, -0.8)
		else:
			return self._beta2

	def get_config(self):
		config = {
			'lr': self._lr,
			'beta1': self.beta1,
			'beta2': self._beta2,
			'epsilon': self.epsilon,
			'epsilon_param_scale': self.epsilon_param_scale,
			'decay_var_list': self.decay_var_list,
			'multiply_by_parameter_scale': self.multiply_by_parameter_scale,
			'clipping_threshold': self.clipping_threshold,
		}
		base_config = super(AdamWBase, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class AdamW(AdamWBase):
	"""AdamW optmizer with beta2 schedule, parameter scaling, and update clipping (tf.keras version)
	Link to paper: https://arxiv.org/abs/1711.05101
	"""
	def __init__(self, *args, **kwargs):
		if 'name' not in kwargs:
			kwargs['name'] = 'AdamW'
		super(AdamW, self).__init__(*args, **kwargs)

	def _create_slots(self, var_list):
		for var in var_list:
			if self.beta1 > 0.0:
				self.add_slot(var, 'm')
			self.add_slot(var, 'v')

	def _apply_weight_decays(self, u):
		norm = K.cast(K.sqrt(self.batch_size / self.iterations), K.floatx())
		if l1 != 0 and l2 != 0:
			decay = l1 * K.sign(var) + l2 * var
		elif l1 != 0:
			decay = l1 * K.sign(var)
		else:
			decay = l2 * var
		u = u - norm * decay
		return u

	def _resource_apply_dense(self, grad, var):
		g2 = K.square(grad) +K.square(K.epsilon())
		v = self.get_slot(var, 'v')
		# Define aux variable
		v_t = self.beta2 * v + (1.0 - self.beta2) * g2
		v_t = K.update(v, v_t)
		# Define main update
		u = grad / (K.sqrt(v_t + self.epsilon) +K.epsilon())
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
			u = u * K.maximum(K.mean(K.sum(K.square(var))), self.epsilon_param_scale)
		# Weight decay
		if var.name in self.decay_var_list.keys():
			l1, l2 = self.decay_var_list[var.name]
			if l1 != 0 or l2 != 0:
				u = self._apply_weight_decays(u)
		# Update parameters
		return K.update(var, var - self.lr * u)

	def _resource_apply_sparse(self, grad, var, indices):
		grad = tf.IndexedSlices(grad, indices, K.shape(var))
		grad = tf.convert_to_tensor(grad)
		return self._resource_apply_dense(grad, var)

def get_weight_decays(model):
	wd_dict = {}
	for layer in model.layers:
		layer_penalties = _get_layer_penalties(layer)
		if layer_penalties:
			for p in layer_penalties:
				weight_name, weight_penalty = p
				if not all(wp == 0 for wp in weight_penalty):
					wd_dict.update({weight_name: weight_penalty})
	return wd_dict

def _get_layer_penalties(layer):
	if hasattr(layer, 'cell') or \
	  (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
		return _rnn_penalties(layer)
	elif hasattr(layer, 'layer') and not hasattr(layer.layer, 'cell'):
		layer = layer.layer

	penalties = []
	for weight_name in ['kernel', 'bias']:
		_lambda = getattr(layer, weight_name + '_regularizer', None)
		if _lambda is not None:
			l1l2 = (float(_lambda.l1), float(_lambda.l2))
			penalties.append([getattr(layer, weight_name).name, l1l2])
			_lambda.l1 = np.array(0., dtype=_lambda.l1.dtype)
			_lambda.l2 = np.array(0., dtype=_lambda.l2.dtype)
	return penalties

def _rnn_penalties(layer):
	penalties = []
	if hasattr(layer, 'backward_layer'):
		for layer in [layer.forward_layer, layer.backward_layer]:
			penalties += _cell_penalties(layer.cell)
		return penalties
	else:
		return _cell_penalties(layer.cell)

def _cell_penalties(rnn_cell):
	cell = rnn_cell
	penalties = []  # kernel-recurrent-bias

	for weight_idx, weight_type in enumerate(['kernel', 'recurrent', 'bias']):
		_lambda = getattr(cell, weight_type + '_regularizer', None)
		if _lambda is not None:
			weight_name = cell.weights[weight_idx].name
			l1l2 = (float(_lambda.l1), float(_lambda.l2))
			penalties.append([weight_name, l1l2])
			_lambda.l1 = np.array(0., dtype=_lambda.l1.dtype)
			_lambda.l2 = np.array(0., dtype=_lambda.l2.dtype)
	return penalties
