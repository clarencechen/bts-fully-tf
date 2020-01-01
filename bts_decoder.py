from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from custom_layers import LocalPlanarGuidance

class BTSDecoder(object):
	def __init__(self, width, height, max_depth, num_filters=256, is_training=False):
		self.num_filters = num_filters
		self.batch_norm_params = {'momentum': 0.99,
						 'epsilon': 1.1e-5,
						 'fused': True, }
		self.full_width = width
		self.full_height = height
		self.max_depth = max_depth
		self.is_training = is_training

	def lpg_block(self, inputs, upratio):
		depth = LocalPlanarGuidance(height=self.full_height, width=self.full_width, upratio=upratio)(inputs)
		return depth

	def conv_block_no_lpg(self, inputs, skips, num_filters):
		upsample = layers.UpSampling2D(size=2, interpolation='nearest')(inputs)
		upconv = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(upsample)
		upconv = layers.BatchNormalization(**self.batch_norm_params)(upconv, training=self.is_training)
		concat = layers.Concatenate(axis=3)([upconv, skips])
		iconv = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(concat)
		return iconv

	def conv_block(self, inputs, skips, lpgs, num_filters):
		upsample = layers.UpSampling2D(size=2, interpolation='nearest')(inputs)
		upconv = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(upsample)
		upconv = layers.BatchNormalization(**self.batch_norm_params)(upconv, training=self.is_training)
		concat = layers.Concatenate(axis=3)([upconv, skips, lpgs])
		iconv = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(concat)
		return iconv

	def dense_aspp_block(self, inputs, num_filters, rate, batch_norm_first=True):
		if batch_norm_first:
			inputs = layers.BatchNormalization(**self.batch_norm_params)(inputs, training=self.is_training)
		relu1 = layers.ReLU()(inputs)
		iconv = layers.Conv2D(num_filters, kernel_size=1, strides=1, padding='same', activation=None)(relu1)
		iconv = layers.BatchNormalization(**self.batch_norm_params)(iconv, training=self.is_training)
		relu2 = layers.ReLU()(iconv)
		daspp = layers.Conv2D(num_filters // 2, kernel_size=3, dilation_rate=rate, padding='same', activation=None)(relu2)
		return daspp

	def build(self, dense_features, skip_2, skip_4, skip_8, skip_16):
		num_filters = self.num_filters

		iconv5 = self.conv_block_no_lpg(dense_features, skip_16, num_filters) # H/16

		num_filters = num_filters // 2
		iconv4 = self.conv_block_no_lpg(iconv5, skip_8, num_filters) # H/8

		iconv4_bn = layers.BatchNormalization(**self.batch_norm_params)(iconv4, training=self.is_training)
		daspp_3 = self.dense_aspp_block(iconv4_bn, num_filters, rate=3, batch_norm_first=False)

		concat4_2 = layers.Concatenate(axis=3)([iconv4, daspp_3]) # Minor edit
		daspp_6 = self.dense_aspp_block(concat4_2, num_filters, rate=6)

		concat4_3 = layers.Concatenate(axis=3)([concat4_2, daspp_6])
		daspp_12 = self.dense_aspp_block(concat4_3, num_filters, rate=12)

		concat4_4 = layers.Concatenate(axis=3)([concat4_3, daspp_12])
		daspp_18 = self.dense_aspp_block(concat4_4, num_filters, rate=18)

		concat4_5 = layers.Concatenate(axis=3)([concat4_4, daspp_18])
		daspp_24 = self.dense_aspp_block(concat4_5, num_filters, rate=24)

		concat4_daspp = layers.Concatenate(axis=3)([iconv4_bn, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24])
		daspp_feat = layers.Conv2D(num_filters // 2, kernel_size=3, strides=1, padding='same', activation='elu')(concat4_daspp)
		daspp_feat = layers.Lambda(lambda t: tf.debugging.assert_all_finite(t, message='daspp_feat tensor is invalid'))(daspp_feat)

		depth_8x8_scaled = self.lpg_block(daspp_feat, upratio=8)
		depth_8x8_scaled_ds = layers.AveragePooling2D(pool_size=4)(depth_8x8_scaled) # Need to find non-trainable Downsampling Layer
		depth_8x8_scaled_ds = layers.Lambda(lambda t: tf.debugging.assert_all_finite(t, message='depth_8x8_scaled_ds tensor is invalid'))(depth_8x8_scaled_ds)

		num_filters = num_filters // 2
		iconv3 = self.conv_block(daspp_feat, skip_4, depth_8x8_scaled_ds, num_filters) # H/4		

		depth_4x4_scaled = self.lpg_block(iconv3, upratio=4)
		depth_4x4_scaled_ds = layers.AveragePooling2D(pool_size=2)(depth_4x4_scaled) # Need to find non-trainable Downsampling Layer
		depth_4x4_scaled_ds = layers.Lambda(lambda t: tf.debugging.assert_all_finite(t, message='depth_4x4_scaled_ds tensor is invalid'))(depth_4x4_scaled_ds)

		num_filters = num_filters // 2
		iconv2 = self.conv_block(iconv3, skip_2, depth_4x4_scaled_ds, num_filters) # H/2

		depth_2x2_scaled = self.lpg_block(iconv2, upratio=2)

		num_filters = num_filters // 2
		upsample1 = layers.UpSampling2D(size=2, interpolation='nearest')(iconv2)
		upconv1 = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(upsample1) # H
		concat1 = layers.Concatenate(axis=3)([upconv1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled])
		iconv1 = layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same', activation='elu')(concat1)
		iconv1 = layers.Lambda(lambda t: tf.debugging.assert_all_finite(t, message='iconv1 tensor is invalid'))(iconv1)

		depth_est_scaled = layers.Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(iconv1)
		depth_est = layers.Lambda(lambda inputs: inputs * self.max_depth)(depth_est_scaled)
		return depth_est
