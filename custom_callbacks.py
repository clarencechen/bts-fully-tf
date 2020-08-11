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

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2

from tensorflow.keras import backend as K
from tensorflow.keras import callbacks

class BatchLRScheduler(callbacks.Callback):
	"""Learning rate scheduler.
	# Arguments
		schedule: a function that takes a batch index as input
			(integer, indexed from 0) and current learning rate
			and returns a new learning rate as output (float).
		verbose: int. 0: quiet, 1: update messages.
	"""

	def __init__(self, schedule, steps_per_epoch, initial_epoch=0, verbose=0):
		super(BatchLRScheduler, self).__init__()
		self.global_step = float(steps_per_epoch * initial_epoch)
		self.schedule = schedule
		self.verbose = verbose

	def on_batch_begin(self, batch, logs=None):
		if not hasattr(self.model.optimizer, 'lr'):
			raise ValueError('Optimizer must have a "lr" attribute.')
		lr = float(K.get_value(self.model.optimizer.lr))
		lr = self.schedule(self.global_step, lr)
		if not isinstance(lr, (float, np.float32, np.float64)):
			raise ValueError('The output of the "schedule" function '
							 'should be float.')
		K.set_value(self.model.optimizer.lr, lr)
		self.global_step += 1

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		logs['lr'] = K.get_value(self.model.optimizer.lr)
		if self.verbose > 0:
			print('\nEpoch %05d: BatchLRScheduler set learning '
				  'rate to %s.' % (epoch + 1, logs['lr']))

class TensorBoardPlusDepthImages(callbacks.TensorBoard):

	def __init__(self, height, width, max_depth, test_image, **kwargs):
		super(TensorboardPlusDepthImages, self).__init__(**kwargs)
		self.image_height = height
		self.image_width = width
		self.est_max_depth = max_depth
		self.test_image = test_image

	def on_epoch_end(self, epoch, logs=None):		
		super(TensorboardPlusDepthImages, self).on_epoch_end(epoch, logs)
		writer = self._get_writer(self._train_run_name)
		with writer.as_default(), summary_ops_v2.always_record_summaries():
			d1_layer = self.model.get_layer(name="depth_est")
			d2_layer = self.model.get_layer(name="depth_2x2_scaled")
			d4_layer = self.model.get_layer(name="depth_4x4_scaled")
			d8_layer = self.model.get_layer(name="depth_8x8_scaled")

			out_map_graph = K.function([self.model.layers[0].input], [d1_layer.output, d2_layer.output, d4_layer.output, d8_layer.output])
			d1, d2_scaled, d4_scaled, d8_scaled = out_map_graph([self.test_image])
			summary_ops_v2.image('input_image', test_image, step=epoch)
			summary_ops_v2.image('depth_est', 1 / (d1 +K.epsilon()), step=epoch)
			summary_ops_v2.image('depth_est_cropped', 1 / (d1[:, 8:self.image_height -8, 8:self.image_width -8, :] +K.epsilon()), step=epoch)
			summary_ops_v2.image('depth_est_2x2', 1 / (d2_scaled * self.est_max_depth + K.epsilon()), step=epoch)
			summary_ops_v2.image('depth_est_4x4', 1 / (d4_scaled * self.est_max_depth + K.epsilon()), step=epoch)
			summary_ops_v2.image('depth_est_8x8', 1 / (d8_scaled * self.est_max_depth + K.epsilon()), step=epoch)