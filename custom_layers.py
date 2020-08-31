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
from math import pi
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class LocalPlanarGuidance(layers.Layer):
	def __init__(self, upratio, **kwargs):
		super(LocalPlanarGuidance, self).__init__(**kwargs)
		self.upratio = upratio

	def build(self, input_shape):
		assert len(input_shape) > 2
		height, width = input_shape[1] * self.upratio, input_shape[2] * self.upratio
		v, u = tf.meshgrid(
			np.linspace(0, width -1, width, dtype=np.float32), 
			np.linspace(0, height -1, height, dtype=np.float32))

		v = K.expand_dims(v, axis=0)
		v = (v % self.upratio - (self.upratio - 1)/2) / float(self.upratio) # V2 Update

		u = K.expand_dims(u, axis=0)
		u = (u % self.upratio - (self.upratio - 1)/2) / float(self.upratio) # V2 Update

		self.pixel_dir_unit = K.l2_normalize(K.stack([u, v, K.ones_like(u)], axis=-1), axis=3) # Normalize first

		return super(LocalPlanarGuidance, self).build(input_shape)

	def call(self, inputs):
		# Decrease max value of theta to pi/3 to enhance numerical stability
		phi, theta, raw_dist = inputs[:, :, :, 0:1] * 2 * pi, inputs[:, :, :, 1:2] * pi / 3, inputs[:, :, :, 2:3]
		plane_coeffs = K.concatenate([K.sin(theta) * K.cos(phi), K.sin(theta) * K.sin(phi), K.cos(theta), raw_dist], axis=-1)
		
		plane_exp_height = K.repeat_elements(plane_coeffs, self.upratio, axis=1)
		plane_exp = K.repeat_elements(plane_exp_height, self.upratio, axis=2)

		denominator = K.sum(self.pixel_dir_unit*plane_exp[..., 0:3], axis=-1, keepdims=True) + K.epsilon()
		return plane_exp[..., 3:] / denominator

	def get_config(self):
		base_config = super(LocalPlanarGuidance, self).get_config()
		config = {'upratio': self.upratio}
		return dict(list(base_config.items()) + list(config.items()))
