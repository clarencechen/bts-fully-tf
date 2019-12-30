from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tf.keras import Model
from tf.keras import backend as K
from tf.keras import layers

class DenseNet(object):
	def __init__(self, nb_layers, growth_rate=48, nb_filter=96, reduction=0.0, dropout_rate=0.0, reg_weight=1e-4, is_training=False):
		'''Instantiate the DenseNet architecture,
		# Arguments
			nb_layers: number of layers to add for each dense block
			growth_rate: number of filters to add per dense block
			nb_filter: initial number of filters
			reduction: reduction factor of transition blocks.
			dropout_rate: dropout rate
			reg_weight: l2 regularization weight factor
			weights_path: path to pre-trained weights
		# Returns
			Outputs of the newly created/loaded model instance.
		'''
		self.nb_layers = nb_layers
		self.growth_rate = growth_rate
		self.nb_dense_block = len(nb_layers)
		self.dropout_rate = dropout_rate
		self.l2_reg = regularizers.l2(reg_weight)
		self.is_training = is_training
		self.compression = 1.0 - reduction
		self.batch_norm_params = {'scale': True,
						 'momentum': 0.99,
						 'epsilon': 1.1e-5,
						 'fused': True}

	def conv_block(self, x, stage, branch, nb_filter, frozen=False):
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
		x = layers.BatchNormalization(name=conv_name_base+'_x1_bn', **self.batch_norm_params, trainable=(not frozen))(x, training=self.is_training)
		x = layers.ReLU(name=relu_name_base+'_x1')(x)
		x = layers.Conv2D(inter_channel, 
						kernel_size=1, 
						padding='same', 
						name=conv_name_base+'_x1', 
						use_bias=False, 
						kernel_regularizer=self.l2_reg, 
						trainable=(not frozen))(x)

		if self.dropout_rate:
			x = layers.Dropout(self.dropout_rate)(x, training=self.is_training)

		# 3x3 Convolution
		x = layers.BatchNormalization(name=conv_name_base+'_x2_bn', **self.batch_norm_params, trainable=(not frozen))(x, training=self.is_training)
		x = layers.ReLU(name=relu_name_base+'_x2')(x)
		x = layers.Conv2D(nb_filter, 
						kernel_size=3, 
						padding='same', 
						name=conv_name_base+'_x2', 
						use_bias=False, 
						kernel_regularizer=self.l2_reg, 
						trainable=(not frozen))(x)

		if self.dropout_rate:
			x = layers.Dropout(self.dropout_rate)(x, training=self.is_training)

	return x

	def transition_block(self, x, stage, nb_filter, frozen=False):
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

		x = layers.BatchNormalization(name=conv_name_base+'_bn', **self.batch_norm_params, trainable=(not frozen))(x, training=is_training)
		x = layers.ReLU(name=relu_name_base)(x)
		x = layers.Conv2D(int(nb_filter * self.compression),
						kernel_size=1, 
						padding='same', 
						name=conv_name_base, 
						use_bias=False, 
						kernel_regularizer=self.l2_reg, 
						trainable=(not frozen))(x)

		if self.dropout_rate:
			x = layers.Dropout(self.dropout_rate)(x, training=is_training)

		x = layers.AveragePooling2D(2, name=pool_name_base)(x)

		return x

	def dense_block(self, x, stage, nb_layers, grow_nb_filters=True, frozen=False):
		''' Build a dense_block where the output of each conv_block is fed to subsequent ones
			# Arguments
				x: input tensor
				stage: index for dense block
				nb_layers: the number of layers of conv_block to append to the model.
				grow_nb_filters: flag to decide to allow number of filters to grow
				frozen: flag to freeze all layers in block
		'''

		eps = 1.1e-5
		concat_feat = x

		for i in range(nb_layers):
			branch = i+1
			x = self.conv_block(concat_feat, stage, branch, self.growth_rate, frozen)
			concat_name = 'concat_'+str(stage)+'_'+str(branch)
			concat_feat = layers.Concatenate(axis=concat_axis, name=concat_name)([concat_feat, x])

			if grow_nb_filters:
				self.nb_filter += self.growth_rate

		return concat_feat

	def build(inputs, weights_path=None, fix_first=False, fix_first_two=False):
		'''Instantiate the DenseNet architecture,
		# Arguments
			weights_path: path to pre-trained weights
		# Returns
			Outputs of the newly created/loaded model instance.
		'''

		skips = []
		
		# Initial convolution
		conv1 = layers.Conv2D(self.nb_filter, kernel_size=7, strides=2, padding='same', name='conv1', bias=False, kernel_regularizer=self.l2_reg)(inputs)
		conv1_bn = layers.BatchNormalization(name='conv1_bn', **self.batch_norm_params)(conv1, training=self.is_training)
		skips.append(conv1_bn)

		relu1 = layers.ReLU(name='relu1')(conv1_bn)
		maxpool1 = layers.MaxPool2D(3, strides=2, padding='same', name='pool1')(relu1)
		skips.append(maxpool1)

		# Add dense blocks
		for block_idx in range(self.nb_dense_block - 1):
			if fix_first_two:
				freeze_layers = (block_idx < 2)
			elif fix_first:
				freeze_layers = (block_idx < 1)
			else:
				freeze_layers = False

			stage = block_idx +2
			x = self.dense_block(x, stage, self.nb_layers[block_idx], frozen=freeze_layers)

			# Add transition_block
			x = self.transition_block(x, stage, self.nb_filter, frozen=freeze_layers)
			self.nb_filter = int(self.nb_filter * self.compression)
			if block_idx < self.nb_dense_block - 2:
				skips.append(x)

		final_stage = stage + 1
		convfinal = self.dense_block(x, final_stage, self.nb_layers[-1], frozen=False)

		convfinal_bn = layers.BatchNormalization(name='conv'+str(final_stage)+'_blk_bn', **self.batch_norm_params)(convfinal, training=self.is_training)
		dense_features = layers.ReLU(name='relu'+str(final_stage)+'_blk')(convfinal_bn)

		outputs = [dense_features] + skips

		# Build model in order to load weights if necessary
		model = Model(inputs, outputs, name='densenet')

		if (weights_path is not None) and (weights_path != ''):
		  model.load_weights(weights_path)

		assert len(outputs) == (1 + self.nb_dense_block)
		return tuple(outputs)


