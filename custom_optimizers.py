# Copyright (C) 2020 Clarence Chen
#
# This file is a part of BTS for Tensorflow 2 with Keras.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class AdamW(tf.keras.optimizers.Adam):
	r"""Implements AdamW algorithm.

	The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
	The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

	Arguments:
		decay_var_list (dict, optional): dictionary mapping weight names to 
			two-element tuples of L1 and L2 weight decay coefficients in that order
			(default: {})

	.. _Adam\: A Method for Stochastic Optimization:
		https://arxiv.org/abs/1412.6980
	.. _Decoupled Weight Decay Regularization:
		https://arxiv.org/abs/1711.05101
	.. _On the Convergence of Adam and Beyond:
		https://openreview.net/forum?id=ryQu7f-RZ
	"""

	def __init__(self, decay_var_list=None, name='AdamW', **kwargs):
		super(AdamW, self).__init__(name=name, **kwargs)
		self.decay_var_list = decay_var_list or {}

	def _compute_weight_decays(self, var):
		l1, l2 = self.decay_var_list[var.name]
		if l1 != 0 and l2 != 0:
			decay = l1 * K.sign(var) + l2 * var
		elif l1 != 0:
			decay = l1 * K.sign(var)
		else:
			decay = l2 * var
		return decay

	def _resource_apply_dense(self, grad, var):
		if var.name in self.decay_var_list.keys():
			K.update(var, var - self.lr * self._compute_weight_decays(var))
		return super(AdamW, self)._resource_apply_dense(grad, var)

	def _resource_apply_sparse(self, grad, var, indices):
		if var.name in self.decay_var_list.keys():
			K.update(var, var - self.lr * self._compute_weight_decays(var))
		return super(AdamW, self)._resource_apply_sparse(grad, var, indices)

	def get_config(self):
		config = {
			'decay_var_list': self.decay_var_list
		}
		base_config = super(AdamW, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def get_weight_decays(model):
	wd_dict = {}
	for layer in model.layers:
		layer_p_dict = _get_layer_penalties(layer)
		if layer_p_dict:
			for weight_name, weight_penalty in layer_p_dict.items():
				if not all(wp == 0 for wp in weight_penalty):
					wd_dict[weight_name] = weight_penalty
	return wd_dict

def _get_layer_penalties(layer):
	if hasattr(layer, 'cell') or (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
		return _rnn_penalties(layer)
	elif hasattr(layer, 'layer') and not hasattr(layer.layer, 'cell'):
		layer = layer.layer

	penalties = {}
	for weight_name in ['kernel', 'bias']:
		_lambda = getattr(layer, weight_name + '_regularizer', None)
		if _lambda is not None:
			penalties[getattr(layer, weight_name).name] = (float(_lambda.l1), float(_lambda.l2))
			_lambda.l1 = np.array(0., dtype=_lambda.l1.dtype)
			_lambda.l2 = np.array(0., dtype=_lambda.l2.dtype)
	return penalties

def _rnn_penalties(layer):
	penalties = []
	if hasattr(layer, 'backward_layer'):
		for layer in [layer.forward_layer, layer.backward_layer]:
			for name, l1l2 in _cell_penalties(layer.cell).items():
				penalties[name] = l1l2
		return penalties
	else:
		return _cell_penalties(layer.cell)

def _cell_penalties(rnn_cell):
	cell = rnn_cell
	penalties = {}  # kernel-recurrent-bias
	for weight_idx, weight_type in enumerate(['kernel', 'recurrent', 'bias']):
		_lambda = getattr(cell, weight_type + '_regularizer', None)
		if _lambda is not None:
			weight_name = cell.weights[weight_idx].name
			penalties[weight_name] = (float(_lambda.l1), float(_lambda.l2))
			_lambda.l1 = np.array(0., dtype=_lambda.l1.dtype)
			_lambda.l2 = np.array(0., dtype=_lambda.l2.dtype)
	return penalties
