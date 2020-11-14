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
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.imagenet_utils import preprocess_input

class BtsReader(object):
	"""Image reader from list of filenames"""

	def __init__(self, params):
		self.params = params

	def read_from_image_files(self, data_path, gt_path, filenames_file, mode='train', shuffle_dataset=True):
		
		def read_decode_predict(line):
			split_line = tf.strings.split([line]).values
			image_path = tf.strings.join([data_path, split_line[0]])
			im_data = tf.io.read_file(image_path)
			image = tf.io.decode_image(im_data, channels=3, expand_animations=False)
			#if self.params.dataset == 'kitti':
			#	focal = tf.strings.to_number(split_line[1])
				# Normalize ground truth depth map to focal length of each example in KITTI Eigen train set
			#	depth_gt /= focal / 715.0873

			image = tf.image.convert_image_dtype(image, tf.float32)
			return image

		def read_decode_test(line):
			split_line = tf.strings.split([line]).values
			image_path = tf.strings.join([data_path, split_line[0]])
			depth_gt_path = tf.strings.join([gt_path, split_line[1]])
			im_data, depth_data = tf.io.read_file(image_path), tf.io.read_file(depth_gt_path)
			image = tf.io.decode_image(im_data, channels=3, expand_animations=False)
			depth_gt = tf.image.decode_png(depth_data, channels=0, dtype=tf.uint16)

			if self.params.dataset == 'nyu':
				depth_gt = tf.cast(depth_gt, tf.float32) / 1000.0
			else:
				depth_gt = tf.cast(depth_gt, tf.float32) / 256.0
			if self.params.dataset == 'kitti':
				focal = tf.strings.to_number(split_line[2])
				# Normalize ground truth depth map to focal length of each example in KITTI Eigen train set
				depth_gt /= focal / 715.0873

			image = tf.image.convert_image_dtype(image, tf.float32)
			return tf.concat([image, depth_gt], 2)

		def read_decode_train(line):
			split_line = tf.strings.split([line]).values
			image_path = tf.strings.join([data_path, split_line[0]])
			depth_gt_path = tf.strings.join([gt_path, split_line[1]])
			im_data, depth_data = tf.io.read_file(image_path), tf.io.read_file(depth_gt_path)
			depth_gt = tf.image.decode_png(depth_data, channels=0, dtype=tf.uint16)

			if self.params.dataset == 'nyu':
				# To avoid blank boundaries due to pixel registration
				image = tf.image.decode_and_crop_jpeg(im_data, [45, 43, 427, 565], channels=3)
				depth_gt = tf.cast(depth_gt[45:472, 43:608, :], tf.float32) / 1000.0
			else:
				image = tf.image.decode_png(im_data, channels=3)
				depth_gt = tf.cast(depth_gt, tf.float32) / 256.0
			if self.params.dataset == 'kitti':
				# Normalize ground truth depth map to focal length of each example in KITTI Eigen training set
				focal = tf.strings.to_number(split_line[2])
				depth_gt /= focal / 715.0873

			image = tf.image.convert_image_dtype(image, tf.float32)
			return tf.concat([image, depth_gt], 2)

		assert not self.params.use_tpu, 'Please use read_from_tf_record() if training model on TPU.'

		with tf.io.gfile.GFile(filenames_file, 'r') as f:
			filenames = f.readlines()
		loader = tf.data.Dataset.from_tensor_slices(filenames)

		if mode == 'train':
			if shuffle_dataset:
				loader = loader.repeat().shuffle(len(filenames))
			loader = loader.map(read_decode_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		elif mode == 'test':
			loader = loader.map(read_decode_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		else:
			loader = loader.map(read_decode_predict, num_parallel_calls=tf.data.experimental.AUTOTUNE)

		return loader

	def read_from_tf_record(self, tf_record_file, filenames_file, mode='train', num_shards=1):
		with tf.io.gfile.GFile(filenames_file, 'r') as f:
			num_examples = len(f.readlines())
		if num_shards > 1:
			loader = tf.data.TFRecordDataset(['{}_part_{}'.format(tf_record_file, i) for i in range(num_shards)],
				compression_type="GZIP",
				num_parallel_reads=tf.data.experimental.AUTOTUNE)
		else:
			loader = tf.data.TFRecordDataset([tf_record_file],
				compression_type="GZIP",
				num_parallel_reads=tf.data.experimental.AUTOTUNE)

		loader = loader.map(lambda x: tf.io.parse_tensor(x, tf.float32), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		if mode == 'train':
			loader = loader.repeat().shuffle(num_examples)

		return loader


class BtsDataloader(object):
	"""bts dataloader"""

	def __init__(self, params, do_rotate=False, degree=5.0, do_kb_crop=False, use_tpu=False):
		self.batch_size = params.batch_size
		self.height = params.height
		self.width = params.width
		self.dataset = params.dataset
		self.encoder = params.encoder
		self.tpu = params.use_tpu

		self.image_mean = tf.reshape(tf.constant([123.68, 116.78, 103.94], dtype=tf.float32), [1, 1, 1, 3])

		self.do_rotate = do_rotate
		self.degree = degree
		self.do_kb_crop = do_kb_crop

	def kb_crop(self, image):
		height, width = tf.shape(image)[0], tf.shape(image)[1]
		top_margin = tf.cast(height - 352, dtype='int32')
		left_margin = tf.cast((width - 1216) / 2, dtype='int32')
		image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

		return image

	def predict_preprocess(self, batch):
		batch.set_shape([self.batch_size, self.height, self.width, 3])

		batch *= 255.0
		if self.encoder == 'densenet161_bts':
			batch = (batch - self.image_mean) * 0.017
		elif 'densenet' in self.encoder:
			batch = preprocess_input(batch, mode='torch')
		else:
			batch = preprocess_input(batch, mode='tf')

		return batch

	def test_preprocess(self, batch):
		image_batch, depth_gt_batch = batch[..., 0:3], batch[..., 3:4]

		image_batch.set_shape([self.batch_size, self.height, self.width, 3])
		depth_gt_batch.set_shape([self.batch_size, self.height, self.width, 1])

		image_batch *= 255.0
		if self.encoder == 'densenet161_bts':
			image_batch = (image_batch - self.image_mean) * 0.017
		elif 'densenet' in self.encoder:
			image_batch = preprocess_input(image_batch, mode='torch')
		else:
			image_batch = preprocess_input(image_batch, mode='tf')

		return image_batch, depth_gt_batch

	def dataset_rotate(self, sample):
		random_angle = tf.random.uniform([], - self.degree * 3.141592 / 180, self.degree * 3.141592 / 180)
		rot_image = tfa.image.rotate(sample[..., 0:3], random_angle, interpolation='BILINEAR')
		rot_gt_depth = tfa.image.rotate(sample[..., 3:4], random_angle, interpolation='NEAREST')
		return tf.concat([rot_image, rot_gt_depth], axis=2)

	def dataset_crop_train(self, sample):
		# Random cropping
		return tf.image.random_crop(sample, [self.height, self.width, 4])

	def train_preprocess(self, batch):
		# Random flipping
		batch = tf.image.random_flip_left_right(batch)

		image_batch, depth_gt_batch = batch[..., 0:3], batch[..., 3:4]

		# Random gamma, brightness, color augmentation
		gamma = tf.random.uniform([self.batch_size, 1, 1, 1], 0.9, 1.1)
		if self.dataset == 'nyu':
			brightness = tf.random.uniform([self.batch_size, 1, 1, 1], 0.75, 1.25)
		else:
			brightness = tf.random.uniform([self.batch_size, 1, 1, 1], 0.9, 1.1)
		colors = tf.random.uniform([self.batch_size, 1, 1, 3], 0.9, 1.1)
		do_augment = tf.random.uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
		augmented = tf.clip_by_value((brightness * colors) * (image_batch ** gamma), 0, 1)
		image_batch = tf.where(do_augment > 0.5, augmented, image_batch)

		image_batch.set_shape([self.batch_size, self.height, self.width, 3])
		depth_gt_batch.set_shape([self.batch_size, self.height, self.width, 1])

		image_batch *= 255.0
		if self.encoder == 'densenet161_bts':
			image_batch = (image_batch - self.image_mean) * 0.017
		elif 'densenet' in self.encoder:
			image_batch = preprocess_input(image_batch, mode='torch')
		else:
			image_batch = preprocess_input(image_batch, mode='tf')

		return image_batch, depth_gt_batch

	def process_dataset(self, loader, mode):
		if self.do_kb_crop is True:
			loader = loader.map(self.kb_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

		if mode == 'train':
			# Image rotation is not supported on TPU if using tfa.image.rotate()
			if not self.tpu and self.do_rotate is True:
				loader = loader.map(self.dataset_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			loader = loader.map(self.dataset_crop_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			loader = loader.batch(self.batch_size, drop_remainder=self.tpu)
			loader = loader.map(self.train_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		else:
			loader = loader.batch(self.batch_size, drop_remainder=self.tpu)
			if mode == 'test':
				loader = loader.map(self.test_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			else:
				loader = loader.map(self.predict_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		
		loader = loader.prefetch(tf.data.experimental.AUTOTUNE)
		return loader
