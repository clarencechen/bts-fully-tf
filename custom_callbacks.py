from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

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

	def __init__(self, schedule, verbose=0):
		super(BatchLRScheduler, self).__init__()
		self.schedule = schedule
		self.verbose = verbose

	def on_batch_begin(self, batch, logs=None):
		if not hasattr(self.model.optimizer, 'lr'):
			raise ValueError('Optimizer must have a "lr" attribute.')
		lr = float(K.get_value(self.model.optimizer.lr))
		lr = self.schedule(batch, lr)
		if not isinstance(lr, (float, np.float32, np.float64)):
			raise ValueError('The output of the "schedule" function '
							 'should be float.')
		K.set_value(self.model.optimizer.lr, lr)

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		logs['lr'] = K.get_value(self.model.optimizer.lr)
		if self.verbose > 0:
			print('\nEpoch %05d: BatchLRScheduler set learning '
				  'rate to %s.' % (epoch + 1, logs['lr']))