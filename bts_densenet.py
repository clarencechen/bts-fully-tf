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

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers, regularizers

def densenet_model(inputs, nb_layers, growth_rate=48, init_nb_filter=96, reduction=0.0, dropout_rate=0.0, reg_weight=1e-3, is_training=False, weights_path=None, fix_first=False, fix_first_two=False):
	'''Instantiate the DenseNet architecture,
	# Arguments
		nb_layers: number of layers to add for each dense block
		growth_rate: number of filters to add per dense block
		init_nb_filter: initial number of filters
		reduction: reduction factor of transition blocks.
		dropout_rate: dropout rate
		reg_weight: l2 regularization weight factor
		weights_path: path to pre-trained weights
	# Returns
		Outputs of the newly created/loaded model instance.
	'''
	nb_dense_block = len(nb_layers)
	l2_reg = regularizers.L1L2(reg_weight)
	compression = 1.0 - reduction
	batch_norm_params = {'momentum': 0.99, 'epsilon': 1.1e-5, 'fused': True}

	def conv_block(x, stage, branch, nb_filter, frozen=False):
		'''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and optional dropout
			# Arguments
				x: input tensor 
				stage: index for dense block
				branch: layer index within each dense block
				nb_filter: number of filters
		'''

		conv_name_base = 'conv' + str(stage) + '_' + str(branch)
		relu_name_base = 'relu' + str(stage) + '_' + str(branch)

		# 1x1 Convolution (Bottleneck layer)
		inter_channel = nb_filter * 4
		x = layers.BatchNormalization(name=conv_name_base+'_x1_bn', **batch_norm_params, trainable=(not frozen))(x, training=False)
		x = layers.ReLU(name=relu_name_base+'_x1')(x)
		x = layers.Conv2D(inter_channel, 
						kernel_size=1, 
						padding='same', 
						name=conv_name_base+'_x1', 
						use_bias=False, 
						kernel_regularizer=l2_reg, 
						trainable=(not frozen))(x)

		if dropout_rate:
			x = layers.Dropout(dropout_rate)(x, training=is_training)

		# 3x3 Convolution
		x = layers.BatchNormalization(name=conv_name_base+'_x2_bn', **batch_norm_params, trainable=(not frozen))(x, training=False)
		x = layers.ReLU(name=relu_name_base+'_x2')(x)
		x = layers.Conv2D(nb_filter, 
						kernel_size=3, 
						padding='same', 
						name=conv_name_base+'_x2', 
						use_bias=False, 
						kernel_regularizer=l2_reg, 
						trainable=(not frozen))(x)

		if dropout_rate:
			x = layers.Dropout(dropout_rate)(x, training=is_training)

		return x

	def transition_block(x, stage, nb_filter, frozen=False):
		''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
			# Arguments
				x: input tensor
				stage: index for dense block
				nb_filter: number of filters
				frozen: flag to freeze all layers in block
		'''

		conv_name_base = 'conv' + str(stage) + '_blk'
		relu_name_base = 'relu' + str(stage) + '_blk'
		pool_name_base = 'pool' + str(stage) 

		x = layers.BatchNormalization(name=conv_name_base+'_bn', **batch_norm_params, trainable=(not frozen))(x, training=False)
		x = layers.ReLU(name=relu_name_base)(x)
		x = layers.Conv2D(int(nb_filter * compression),
						kernel_size=1, 
						padding='same', 
						name=conv_name_base, 
						use_bias=False, 
						kernel_regularizer=l2_reg, 
						trainable=(not frozen))(x)

		if dropout_rate:
			x = layers.Dropout(dropout_rate)(x, training=is_training)

		x = layers.AveragePooling2D(2, name=pool_name_base)(x)

		return x

	def dense_block(x, stage, conv_layers, frozen=False):
		''' Build a dense_block where the output of each conv_block is fed to subsequent ones
			# Arguments
				x: input tensor
				stage: index for dense block
				conv_layers: the number of layers of conv_block to append to the model.
				frozen: flag to freeze all layers in block
		'''

		concat_feat = x

		for i in range(conv_layers):
			branch = i+1
			x = conv_block(concat_feat, stage, branch, growth_rate, frozen)
			concat_name = 'concat_'+str(stage)+'_'+str(branch)
			concat_feat = layers.Concatenate(axis=3, name=concat_name)([concat_feat, x])

		return concat_feat

	skips = []
	nb_filter = init_nb_filter
	freeze_init = not (fix_first or fix_first_two)
	# Initial convolution
	conv1 = layers.Conv2D(nb_filter,
						kernel_size=7, 
						strides=2, 
						padding='same', 
						name='conv1', 
						use_bias=False, 
						trainable = freeze_init, 
						kernel_regularizer=l2_reg)(inputs)
	conv1_bn = layers.BatchNormalization(name='conv1_bn', **batch_norm_params, trainable=freeze_init)(conv1, training=False)
	relu1 = layers.ReLU(name='relu1')(conv1_bn)
	skips.append(relu1)

	maxpool1 = layers.MaxPool2D(3, strides=2, padding='same', name='pool1')(relu1)
	skips.append(maxpool1)
	x = maxpool1
	# Add dense blocks
	for block_idx in range(nb_dense_block - 1):
		if fix_first_two:
			freeze_layers = (block_idx < 2)
		elif fix_first:
			freeze_layers = (block_idx < 1)
		else:
			freeze_layers = False

		stage = block_idx +2
		x = dense_block(x, stage, nb_layers[block_idx], frozen=freeze_layers)
		nb_filter += nb_layers[block_idx] * growth_rate

		# Add transition_block
		x = transition_block(x, stage, nb_filter, frozen=freeze_layers)
		nb_filter = int(nb_filter * compression)
		if block_idx < nb_dense_block - 2:
			skips.append(x)

	final_stage = stage + 1
	convfinal = dense_block(x, final_stage, nb_layers[-1], frozen=False)

	convfinal_bn = layers.BatchNormalization(name='conv'+str(final_stage)+'_blk_bn', **batch_norm_params)(convfinal, training=False)
	dense_features = layers.ReLU(name='relu'+str(final_stage)+'_blk')(convfinal_bn)

	outputs = [dense_features] + skips

	# Build model in order to load weights if necessary
	model = Model(inputs, outputs, name='densenet')

	if (weights_path is not None) and (weights_path != ''):
		model.load_weights(weights_path, by_name=True)

	assert len(outputs) == (1 + nb_dense_block)
	return tuple(outputs)
