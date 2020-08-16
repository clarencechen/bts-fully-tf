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
from tensorflow.keras import layers

from custom_layers import LocalPlanarGuidance

def decoder_model(decoder_inputs, max_depth, num_filters=256, is_training=False):
	batch_norm_params = {'momentum': 0.99, 'epsilon': 1.1e-5, 'fused': True}
	dense_features, skip_2, skip_4, skip_8, skip_16 = decoder_inputs

	def conv_block_no_lpg(inputs, skips, num_filters):
		upsample = layers.UpSampling2D(size=2, interpolation='nearest')(inputs)
		upconv = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu', use_bias=False)(upsample)
		upconv = layers.BatchNormalization(**batch_norm_params)(upconv, training=is_training)
		concat = layers.Concatenate(axis=3)([upconv, skips])
		iconv = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu', use_bias=False)(concat)
		return iconv

	def conv_block(inputs, skips, lpgs, num_filters):
		upsample = layers.UpSampling2D(size=2, interpolation='nearest')(inputs)
		upconv = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu', use_bias=False)(upsample)
		upconv = layers.BatchNormalization(**batch_norm_params)(upconv, training=is_training)
		concat = layers.Concatenate(axis=3)([upconv, skips, lpgs])
		iconv = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu', use_bias=False)(concat)
		return iconv

	def dense_aspp_block(inputs, num_filters, rate, batch_norm_first=True):
		if batch_norm_first:
			inputs = layers.BatchNormalization(**batch_norm_params)(inputs, training=is_training)
		relu1 = layers.ReLU()(inputs)
		iconv = layers.Conv2D(num_filters, kernel_size=1, strides=1, padding='same', activation=None, use_bias=False)(relu1)
		iconv = layers.BatchNormalization(**batch_norm_params)(iconv, training=is_training)
		relu2 = layers.ReLU()(iconv)
		daspp = layers.Conv2D(num_filters // 2, kernel_size=3, dilation_rate=rate, padding='same', activation=None, use_bias=False)(relu2)
		return daspp

	iconv5 = conv_block_no_lpg(dense_features, skip_16, num_filters) # H/16

	num_filters = num_filters // 2
	iconv4 = conv_block_no_lpg(iconv5, skip_8, num_filters) # H/8

	iconv4_bn = layers.BatchNormalization(**batch_norm_params)(iconv4, training=is_training)
	daspp_3 = dense_aspp_block(iconv4_bn, num_filters, rate=3, batch_norm_first=False)

	concat4_2 = layers.Concatenate(axis=3)([iconv4, daspp_3]) # Minor edit
	daspp_6 = dense_aspp_block(concat4_2, num_filters, rate=6)

	concat4_3 = layers.Concatenate(axis=3)([concat4_2, daspp_6])
	daspp_12 = dense_aspp_block(concat4_3, num_filters, rate=12)

	concat4_4 = layers.Concatenate(axis=3)([concat4_3, daspp_12])
	daspp_18 = dense_aspp_block(concat4_4, num_filters, rate=18)

	concat4_5 = layers.Concatenate(axis=3)([concat4_4, daspp_18])
	daspp_24 = dense_aspp_block(concat4_5, num_filters, rate=24)

	concat4_daspp = layers.Concatenate(axis=3)([iconv4_bn, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24])
	daspp_feat = layers.Conv2D(num_filters // 2, kernel_size=3, strides=1, padding='same', activation='elu', use_bias=False)(concat4_daspp)

	reduction_8x8 = layers.Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid', use_bias=False)(daspp_feat)
	depth_8x8_scaled = LocalPlanarGuidance(upratio=8, name='depth_8x8_scaled')(reduction_8x8)
	depth_8x8_scaled_ds = layers.Lambda(lambda x : x[:, ::4, ::4, ...])(depth_8x8_scaled) # Downsampling Layer

	num_filters = num_filters // 2
	iconv3 = conv_block(daspp_feat, skip_4, depth_8x8_scaled_ds, num_filters) # H/4		

	reduction_4x4 = layers.Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid', use_bias=False)(iconv3)
	depth_4x4_scaled = LocalPlanarGuidance(upratio=4, name='depth_4x4_scaled')(reduction_4x4)
	depth_4x4_scaled_ds = layers.Lambda(lambda x : x[:, ::2, ::2, ...])(depth_4x4_scaled) # Downsampling Layer

	num_filters = num_filters // 2
	iconv2 = conv_block(iconv3, skip_2, depth_4x4_scaled_ds, num_filters) # H/2

	reduction_2x2 = layers.Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid', use_bias=False)(iconv2)
	depth_2x2_scaled = LocalPlanarGuidance(upratio=2, name='depth_2x2_scaled')(reduction_2x2)

	num_filters = num_filters // 2
	upsample1 = layers.UpSampling2D(size=2, interpolation='nearest')(iconv2)
	upconv1 = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu', use_bias=False)(upsample1) # H
	concat1 = layers.Concatenate(axis=3)([upconv1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled])
	iconv1 = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu', use_bias=False)(concat1)

	depth_est_scaled = layers.Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid', use_bias=False)(iconv1)
	depth_est = layers.Lambda(lambda inputs: inputs * max_depth, name='depth_est')(depth_est_scaled)

	return depth_est