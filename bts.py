# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
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

from tensorflow.keras import Input, Model
from bts_decoder import decoder_model
from bts_densenet import densenet_model

def si_log_loss_wrapper(dataset):
	gt_th = {'nyu':0.1, 'kitti':1.0, 'matterport':0.1}

	def si_log_loss(y_true, y_pred):
		mask = K.greater(y_true, gt_th[dataset])

		y_true_masked = tf.boolean_mask(tensor=y_true, mask=mask)
		y_pred_masked = tf.boolean_mask(tensor=y_pred, mask=mask)

		d = K.log(y_true_masked +K.epsilon()) -K.log(y_pred_masked +K.epsilon())
		return K.sqrt(K.mean(K.square(d)) - 0.85 * K.square(K.mean(d))) * 10.0 # Differs from paper

	assert dataset in gt_th
	return si_log_loss

def bts_model(params, mode, start_lr, fix_first=False, fix_first_two=False, pretrained_weights_path=None):
	
	is_training = True if mode == 'train' else False
	input_image = Input(shape=(params.height, params.width, 3), batch_size=params.batch_size)

	if params.encoder == 'densenet161_bts':
		dense_features, skip_2, skip_4, skip_8, skip_16 = densenet_model(input_image, 
																		 [6,12,36,24],
																		 growth_rate=48,
																		 init_nb_filter=96,
																		 reduction=0.5,
																		 is_training=is_training,
																		 weights_path=pretrained_weights_path,
																		 fix_first=fix_first,
																		 fix_first_two=fix_first_two)
		depth_est = decoder_model(dense_features, skip_2, skip_4, skip_8, skip_16, 
																		  params.height, 
																		  params.width, 
																		  params.max_depth, 
																		  num_filters=512, 
																		  is_training=is_training)
	elif params.encoder == 'densenet121_bts':
		dense_features, skip_2, skip_4, skip_8, skip_16 = densenet_model(input_image,
																		 [6,12,24,16],
																		 growth_rate=32,
																		 init_nb_filter=64,
																		 reduction=0.5,
																		 is_training=is_training,
																		 weights_path=pretrained_weights_path,
																		 fix_first=fix_first,
																		 fix_first_two=fix_first_two)
		depth_est = decoder_model(dense_features, skip_2, skip_4, skip_8, skip_16, 
																		  params.height, 
																		  params.width,
																		  params.max_depth, 
		 																  num_filters=256, 
																		  is_training=is_training)
	else:
		return None

	model = Model(inputs=input_image, outputs=depth_est)
	return model

'''
def build_summaries(self):
	with tf.device('/cpu:0'):
		tf.compat.v1.summary.scalar('silog_loss', self.silog_loss, collections=self.model_collection)

		depth_gt = tf.where(self.depth_gt < 1e-3, self.depth_gt * 0 + 1e3, self.depth_gt)
		tf.compat.v1.summary.image('depth_gt', 1 / depth_gt, max_outputs=4, collections=self.model_collection)
		tf.compat.v1.summary.image('depth_est', 1 / self.depth_est, max_outputs=4, collections=self.model_collection)
		tf.compat.v1.summary.image('depth_est_cropped',
						 1 / self.depth_est[:, 8:params.height - 8, 8:self.params.width - 8, :], max_outputs=4,
						 collections=self.model_collection)
		tf.compat.v1.summary.image('depth_est_2x2', 1 / self.depth_2x2, max_outputs=4, collections=self.model_collection)
		tf.compat.v1.summary.image('depth_est_4x4', 1 / self.depth_4x4, max_outputs=4, collections=self.model_collection)
		tf.compat.v1.summary.image('depth_est_8x8', 1 / self.depth_8x8, max_outputs=4, collections=self.model_collection)
		tf.compat.v1.summary.image('image', self.input_image[:, :, :, ::-1], max_outputs=4, collections=self.model_collection)
'''