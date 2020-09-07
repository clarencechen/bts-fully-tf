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

from tensorflow.keras import Input, Model, applications
from bts_decoder import decoder_model
from bts_densenet import densenet_model

def si_log_loss_wrapper(dataset):
	gt_th = {'nyu':0.1, 'kitti':1.0, 'matterport':0.1}

	@tf.function()
	def si_log_loss(y_true, y_pred):
		mask = K.greater(y_true, gt_th[dataset])

		y_true_masked = tf.boolean_mask(tensor=y_true, mask=mask)
		y_pred_masked = tf.boolean_mask(tensor=y_pred, mask=mask)

		d = K.log(y_true_masked +K.epsilon()) -K.log(y_pred_masked +K.epsilon())
		return K.sqrt(K.mean(K.square(d)) - 0.85 * K.square(K.mean(d))) * 10.0 # Differs from paper

	assert dataset in gt_th
	return si_log_loss

def bts_model(params, mode, fix_first=False, fix_second=False, pretrained_weights_path=None):
	is_training = True if mode == 'train' else False
	input_image = Input(shape=(params.height, params.width, 3), batch_size=params.batch_size, name='input_image')

	if params.encoder == 'densenet161_bts':
		encoder_model = densenet_model(input_image, [6,12,36,24], weights_path=pretrained_weights_path)
		decoder_filters = 512
	elif params.encoder == 'densenet201':
		encoder_model = applications.DenseNet201(include_top=False, input_tensor=input_image, pooling=None)
		decoder_filters = 512
	elif params.encoder == 'densenet169':
		encoder_model = applications.DenseNet169(include_top=False, input_tensor=input_image, pooling=None)
		decoder_filters = 256
	elif params.encoder == 'densenet121':
		encoder_model = applications.DenseNet121(include_top=False, input_tensor=input_image, pooling=None)
		decoder_filters = 256
	elif params.encoder == 'resnet101':
		encoder_model = applications.ResNet101V2(include_top=False, input_tensor=input_image, pooling=None)
		decoder_filters = 512
	elif params.encoder == 'resnet50':
		encoder_model = applications.ResNet50V2(include_top=False, input_tensor=input_image, pooling=None)
		decoder_filters = 256
	elif params.encoder == 'mobilenet':
		encoder_model = applications.MobileNetV2(include_top=False, input_tensor=input_image, pooling=None)
		decoder_filters = 256
	else:
		return None, None

	if 'densenet' in params.encoder:
		out_layers = ['relu', 'conv1/relu', 'pool1', 'pool2_pool', 'pool3_pool']
		fix_list = ['conv1', 'pool1']
		if fix_first:
			fix_list += ['conv2', 'pool2']
			if fix_second:
				fix_list += ['conv3', 'pool3']

	elif 'resnet' in params.encoder:
		out_layers = ['post_relu', 'conv1_conv', 'pool1_pool', 'conv2_block3_out', 'conv3_block4_out']
		fix_list = ['conv1', 'pool1']
		if fix_first:
			fix_list += ['conv2']
			if fix_second:
				fix_list += ['conv3']

	elif 'mobilenet' in params.encoder:
		out_layers = ['out_relu', 'Conv1_relu', 'block_2_add', 'block_5_add', 'block_12_add']
		fix_list = ['Conv1', 'block_0']
		if fix_first:
			fix_list += ['block_1, block_2']
			if fix_second:
				fix_list += ['block_3', 'block_4', 'block_5']

	encoder_outputs = [encoder_model.get_layer(name).output for name in out_layers]
	for layer in encoder_model.layers:
		for substr in fix_list:
			if layer.name.startswith(substr):
				layer.trainable = False

	depth_est = decoder_model(encoder_outputs, 
				params.max_depth, 
				num_filters=decoder_filters, 
				is_training=is_training)

	model = Model(inputs=input_image, outputs=depth_est)
	return model, compile_weight_decays(encoder_model, params.encoder, l1=0.0, l2=1e-3)

def compile_weight_decays(model, model_name, l1=0.0, l2=0.0):
	wd_dict = {}

	for layer in model.layers:
		if hasattr(layer, 'layer'):
			layer = layer.layer
		is_conv_layer = (layer.name in ['Conv1', 'Conv_1'] or layer.name[-6:] in ['expand', 'roject', 'thwise']) if ('mobilenet' in model_name) else (layer.name[-4:] == 'conv')
		if is_conv_layer and hasattr(layer, 'kernel'):
			wd_dict[getattr(layer, 'kernel').name] = (float(l1), float(l2))
	return wd_dict
