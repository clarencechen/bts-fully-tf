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

class TensorboardPlusDepthImages(callbacks.TensorBoard):

	def __init__(self, height, width, max_depth, **kwargs):
		super(TensorboardPlusDepthImages, self).__init__(**kwargs)
		self.image_height = height
		self.image_width = width
		self.est_max_depth = max_depth

	def on_epoch_end(self, epoch, logs=None):		
		super(TensorboardPlusDepthImages, self).on_epoch_end(epoch, logs)
		writer = self._get_writer(self._train_run_name)
		with context.eager_mode(), writer.as_default(), summary_ops_v2.always_record_summaries():
			in_layer = self.model.get_layer(name="input_image")
			d1_layer = self.model.get_layer(name="depth_est")
			d2_layer = self.model.get_layer(name="depth_2x2_scaled")
			d4_layer = self.model.get_layer(name="depth_4x4_scaled")
			d8_layer = self.model.get_layer(name="depth_8x8_scaled")

			with ops.init_scope():
				# in_img = K.get_value(in_layer.weights)
				# d1 = K.get_value(d1_layer.weights[0])
				d2_scaled = K.get_value(d2_layer.weights[0])
				d4_scaled = K.get_value(d4_layer.weights[0])
				d8_scaled = K.get_value(d8_layer.weights[0])

			# summary_ops_v2.image('input_image', in_img[:, :, :, ::-1], step=epoch)
			# summary_ops_v2.image('depth_est', 1 / (d1 +K.epsilon()), step=epoch)
			# summary_ops_v2.image('depth_est_cropped', 1 / (d1[:, 8:self.image_height -8, 8:self.image_width -8, :] +K.epsilon()), step=epoch)
			summary_ops_v2.image('depth_est_2x2', 1 / (d2_scaled * self.est_max_depth + K.epsilon()), step=epoch)
			summary_ops_v2.image('depth_est_4x4', 1 / (d4_scaled * self.est_max_depth + K.epsilon()), step=epoch)
			summary_ops_v2.image('depth_est_8x8', 1 / (d8_scaled * self.est_max_depth + K.epsilon()), step=epoch)