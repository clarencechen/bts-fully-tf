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

import os
import argparse
import sys

from collections import namedtuple

bts_parameters = namedtuple('parameters', 'encoder, '
										  'height, width, '
										  'max_depth, '
										  'batch_size, '
										  'dataset, '
										  'num_threads, '
										  'num_epochs, '
										  'use_tpu, ')

import tensorflow as tf
from bts_dataloader import BtsReader

def convert_arg_line_to_args(arg_line):
	for arg in arg_line.split():
		if not arg.strip():
			continue
		yield arg


parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=False)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--tfrecord_path',             type=str,   help='path to the output TFRecord dataset in zip format', required=False)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)


if sys.argv.__len__() == 2:
	arg_filename_with_prefix = '@' + sys.argv[1]
	args = parser.parse_args([arg_filename_with_prefix])
else:
	args = parser.parse_args()

def get_num_lines(file_path):
	f = open(file_path, 'r')
	lines = f.readlines()
	f.close()
	return len(lines)

def main():
	params = bts_parameters(
		encoder=None,
		height=None,
		width=None,
		batch_size=None,
		dataset=args.dataset,
		max_depth=None,
		num_threads=args.num_threads,
		num_epochs=None,
		use_tpu=False)

	reader = BtsReader(params)
	loader = reader.read_from_image_files(data_path, gt_path, filenames_file, 'train', shuffle_dataset=False)

	loader = loader.map(tf.io.serialize_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
	loader = loader.prefetch(tf.data.experimental.AUTOTUNE)

	writer = tf.data.experimental.TFRecordWriter(tf_record_path, compression_type="GZIP")
	writer.write(loader)

if __name__ == '__main__':
	main()
