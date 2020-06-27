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
from tensorflow.python.ops import array_ops

class BtsReader(object):
	"""Image reader from list of filenames"""

	def __init__(self, params):
		self.params = params

	def read_from_image_files(self, data_path, gt_path, filenames_file, mode='train', shuffle_dataset=True):
		def read_decode_test(line):
			split_line = tf.strings.split([line]).values
			image_path = tf.strings.join([data_path, split_line[0]])
			depth_gt_path = tf.strings.join([gt_path[:-1], split_line[1]])
			im_data, depth_data = tf.io.read_file(image_path), tf.io.read_file(depth_gt_path)
			depth_gt = tf.image.decode_png(depth_data, channels=0, dtype=tf.uint16)

			image = tf.image.decode_jpeg(im_data)
			if self.params.dataset == 'nyu':
				depth_gt = tf.cast(depth_gt, tf.float32) / 1000.0
			else:
				depth_gt = tf.cast(depth_gt, tf.float32) / 256.0

			image = tf.image.convert_image_dtype(image, tf.float32)
			# paper V2 update does not require focal
			#focal = tf.strings.to_number(split_line[2])
			return tf.concat([image, depth_gt], 2)

		def read_decode_train(line):
			split_line = tf.strings.split([line]).values
			image_path = tf.strings.join([data_path, split_line[0]])
			depth_gt_path = tf.strings.join([gt_path, split_line[1]])
			#paper V2 update does not require focal
			#focal = tf.strings.to_number(split_line[2])
			im_data, depth_data = tf.io.read_file(image_path), tf.io.read_file(depth_gt_path)
			depth_gt = tf.image.decode_png(depth_data, channels=0, dtype=tf.uint16)

			if self.params.dataset == 'nyu':
				# To avoid blank boundaries due to pixel registration
				image = tf.image.decode_and_crop_jpeg(im_data, [45, 43, 427, 565])
				depth_gt = tf.cast(depth_gt[45:472, 43:608, :], tf.float32) / 1000.0
			else:
				image = tf.image.decode_png(im_data)
				depth_gt = tf.cast(depth_gt, tf.float32) / 256.0

			image = tf.image.convert_image_dtype(image, tf.float32)
			return tf.concat([image, depth_gt], 2)

		assert not self.params.use_tpu, 'Please use read_from_tf_record() if training model on TPU.'

		with tf.io.gfile.GFile(filenames_file, 'r') as f:
			filenames = f.readlines()
		loader = tf.data.Dataset.from_tensor_slices(filenames)

		if mode == 'train':
			if shuffle_dataset:
				loader = loader.repeat().shuffle(len(filenames))
			loader = loader.map(read_decode_train, num_parallel_calls=self.params.num_threads)
		else:
			loader = loader.map(read_decode_test, num_parallel_calls=self.params.num_threads)

		return loader

	def read_from_tf_record(self, tf_record_file, filenames_file, mode='train', num_shards=1):
		num_data_workers = tf.data.experimental.AUTOTUNE if self.params.use_tpu else self.params.num_threads
		# Change below hardocded value later
		with tf.io.gfile.GFile(filenames_file, 'r') as f:
			num_examples = len(f.readlines())
		if num_shards > 1:
			loader = tf.data.TFRecordDataset(['{}_part_{}'.format(tf_record_file, i) for i in range(num_shards)],
				compression_type="GZIP",
				num_parallel_reads=num_data_workers)
		else:
			loader = tf.data.TFRecordDataset([tf_record_file],
				compression_type="GZIP",
				num_parallel_reads=num_data_workers)
		if mode == 'train':
			loader = loader.repeat().shuffle(num_examples)
			loader = loader.map(lambda x: tf.io.parse_tensor(x, tf.float32), num_parallel_calls=num_data_workers)
		else:
			loader = loader.map(lambda x: tf.io.parse_tensor(x, tf.float32), num_parallel_calls=num_data_workers)
		
		return loader


class BtsDataloader(object):
	"""bts dataloader"""

	def __init__(self, params, do_rotate=False, degree=5.0, do_kb_crop=False, use_tpu=False):
		self.params = params

		self.do_rotate = do_rotate
		self.degree = degree
		self.do_kb_crop = do_kb_crop
		self.image_mean = tf.reshape(tf.constant([123.68, 116.78, 103.94], dtype=tf.float32), [1, 1, 1, 3])

	def dataset_crop_test(self, image):
		height, width = tf.shape(image)[0], tf.shape(image)[1]
		top_margin = tf.cast(height - 352, dtype='int32')
		left_margin = tf.cast((width - 1216) / 2, dtype='int32')
		image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

		return image

	def test_preprocess(self, batch):
		image_batch, depth_gt_batch = batch[..., 0:3], batch[..., 3:4]

		image_batch.set_shape([self.params.batch_size, self.params.height, self.params.width, 3])
		depth_gt_batch.set_shape([self.params.batch_size, self.params.height, self.params.width, 1])

		image_batch = image_batch * 255.0 - self.image_mean

		if self.params.encoder == 'densenet161_bts' or self.params.encoder == 'densenet121_bts':
			image_batch *= 0.017

		return image_batch, depth_gt_batch

	def dataset_crop_train(self, sample):
		if self.do_kb_crop is True:
			height, width = tf.shape(sample)[0], tf.shape(sample)[1]
			top_margin = tf.cast(height - 352, dtype='int32')
			left_margin = tf.cast((width - 1216) / 2, dtype='int32')
			sample = sample[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

		# Image rotation is not supported on TPU if using tfa.image.rotate()
		if not self.params.use_tpu and self.do_rotate is True:
			random_angle = tf.random.uniform([], - self.degree * 3.141592 / 180, self.degree * 3.141592 / 180)
			rot_image = tfa.image.rotate(sample[..., 0:3], random_angle, interpolation='BILINEAR')
			rot_gt_depth = tfa.image.rotate(sample[..., 3:4], random_angle, interpolation='NEAREST')
			sample = tf.concat([rot_image, rot_gt_depth], axis=2)

		# Random cropping
		sample = tf.image.random_crop(sample, [self.params.height, self.params.width, 4])

		return sample

	def train_preprocess(self, batch):
		# Random flipping
		batch = tf.image.random_flip_left_right(batch)

		image_batch, depth_gt_batch = batch[..., 0:3], batch[..., 3:4]

		# Random gamma, brightness, color augmentation
		gamma = tf.random.uniform([self.params.batch_size, 1, 1, 1], 0.9, 1.1)
		if self.params.dataset == 'nyu':
			brightness = tf.random.uniform([self.params.batch_size, 1, 1, 1], 0.75, 1.25)
		else:
			brightness = tf.random.uniform([self.params.batch_size, 1, 1, 1], 0.9, 1.1)
		colors = tf.random.uniform([self.params.batch_size, 1, 1, 3], 0.9, 1.1)
		do_augment = tf.math.round(tf.random.uniform([self.params.batch_size, 1, 1, 1], 0.0, 1.0))
		augmented = tf.clip_by_value((brightness * colors) * (image_batch ** gamma), 0, 1)
		image_batch = do_augment * augmented + (1.0 - do_augment) * image_batch

		image_batch.set_shape([self.params.batch_size, self.params.height, self.params.width, 3])
		depth_gt_batch.set_shape([self.params.batch_size, self.params.height, self.params.width, 1])

		image_batch = image_batch * 255.0 - self.image_mean

		if self.params.encoder == 'densenet161_bts' or self.params.encoder == 'densenet121_bts':
			image_batch *= 0.017

		return image_batch, depth_gt_batch

	def process_dataset(self, loader, mode):
		num_data_workers = tf.data.experimental.AUTOTUNE if self.params.use_tpu else self.params.num_threads

		if mode == 'train':
			loader = loader.map(self.dataset_crop_train, num_parallel_calls=num_data_workers)
			loader = loader.batch(self.params.batch_size, drop_remainder=self.params.use_tpu)
			loader = loader.map(self.train_preprocess, num_parallel_calls=num_data_workers)
			loader = loader.prefetch(tf.data.experimental.AUTOTUNE)
		else:
			if self.do_kb_crop is True:
				loader = loader.map(self.dataset_crop_test, num_parallel_calls=num_data_workers)
			loader = loader.batch(self.params.batch_size, drop_remainder=self.params.use_tpu)
			loader = loader.map(self.test_preprocess, num_parallel_calls=num_data_workers)
			loader = loader.prefetch(tf.data.experimental.AUTOTUNE)
		return loader
