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

def densenet_model(inputs, nb_layers, growth_rate=48, init_nb_filter=96, reduction=0.5, weights_path=None):
	'''Instantiate the DenseNet architecture,
	# Arguments
		nb_layers: number of layers to add for each dense block
		growth_rate: number of filters to add per dense block
		init_nb_filter: initial number of filters
		reduction: reduction factor of transition blocks.
		weights_path: path to pre-trained weights
	# Returns
		Outputs of the newly created/loaded model instance.
	'''
	compression = 1.0 - reduction
	batch_norm_params = {'momentum': 0.99, 'epsilon': 1.1e-5, 'fused': True}

	def conv_block(x, block_name, nb_filter):
		'''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D
			# Arguments
				x: input tensor 
				block_name: name prefix for layers in dense block
				nb_filter: number of filters
		'''

		# 1x1 Convolution (Bottleneck layer)
		x = layers.BatchNormalization(name=block_name+'1_bn', **batch_norm_params)(x, training=False)
		x = layers.ReLU(name=block_name+'1_relu')(x)
		x = layers.Conv2D(nb_filter * 4, 1, padding='same', name=block_name+'1_conv', use_bias=False)(x)

		# 3x3 Convolution
		x = layers.BatchNormalization(name=block_name+'2_bn', **batch_norm_params)(x, training=False)
		x = layers.ReLU(name=block_name+'2_relu')(x)
		x = layers.Conv2D(nb_filter, 3, padding='same', name=block_name+'2_conv', use_bias=False)(x)

		return x

	def transition_block(x, stage, nb_filter):
		''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, 
			# Arguments
				x: input tensor
				stage: index for dense block
				nb_filter: number of filters
		'''


		name_base = 'pool{}_'.format(stage)

		x = layers.BatchNormalization(name=name_base+'bn', **batch_norm_params)(x, training=False)
		x = layers.ReLU(name=name_base+'relu')(x)
		x = layers.Conv2D(int(nb_filter * compression), 1, padding='same', name=name_base+'conv', use_bias=False)(x)
		x = layers.AveragePooling2D(2, name=name_base+'pool')(x)

		return x

	def dense_block(x, stage, conv_layers):
		''' Build a dense_block where the output of each conv_block is fed to subsequent ones
			# Arguments
				x: input tensor
				stage: index for dense block
				conv_layers: the number of layers of conv_block to append to the model.
		'''

		concat_feat = x

		for i in range(conv_layers):
			name_base = 'conv{}_block{}_'.format(stage, i + 1)
			x = conv_block(concat_feat, name_base, growth_rate)
			concat_feat = layers.Concatenate(axis=3, name=name_base+'concat')([concat_feat, x])

		return concat_feat

	nb_filter = init_nb_filter
	# Initial convolution
	conv1 = layers.Conv2D(nb_filter, 7, strides=2, padding='same', name='conv1', use_bias=False)(inputs)
	conv1_bn = layers.BatchNormalization(name='conv1/bn', **batch_norm_params)(conv1, training=False)
	relu1 = layers.ReLU(name='conv1/relu')(conv1_bn)

	maxpool1 = layers.MaxPool2D(3, strides=2, padding='same', name='pool1')(relu1)
	x = maxpool1
	# Add dense blocks
	for block_idx in range(len(nb_layers) - 1):

		stage = block_idx + 2
		x = dense_block(x, stage, nb_layers[block_idx])
		nb_filter += nb_layers[block_idx] * growth_rate

		# Add transition_block
		x = transition_block(x, stage, nb_filter)
		nb_filter = int(nb_filter * compression)

	final_stage = stage + 1
	convfinal = dense_block(x, final_stage, nb_layers[-1])

	convfinal_bn = layers.BatchNormalization(name='bn', **batch_norm_params)(convfinal, training=False)
	dense_out = layers.ReLU(name='relu')(convfinal_bn)

	model = Model(inputs, dense_out, name='densenet_161')
	if (weights_path is not None) and (weights_path != ''):
		model.load_weights(weights_path, by_name=True)

	return model
