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

import os
import numpy as np
import argparse
import tensorflow as tf
import errno
import matplotlib.pyplot as plt
import cv2
import sys
from tqdm import tqdm

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
from tensorflow.keras import callbacks

from bts_dataloader import BtsReader, BtsDataloader
from bts import bts_model

def convert_arg_line_to_args(arg_line):
	for arg in arg_line.split():
		if not arg.strip():
			continue
		yield arg


parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',          type=str,   help='model name', default='bts_nyu_test')
parser.add_argument('--encoder',             type=str,   help='type of encoder, vgg or desenet121_bts or densenet161_bts', default='densenet161_bts')
parser.add_argument('--data_path',           type=str,   help='path to the data')
parser.add_argument('--tfrecord_path',       type=str,   help='path to the combined TFRecord dataset in zip format')
parser.add_argument('--tfrecord_shards',     type=int,   help='number of shards of the combined TFRecord dataset to read', default=1)
parser.add_argument('--filenames_file',      type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',        type=int,   help='input height', default=480)
parser.add_argument('--input_width',         type=int,   help='input width', default=640)
parser.add_argument('--max_depth',           type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path',     type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--batch_size',          type=int,   help='batch size per training replica', default=1)
parser.add_argument('--dataset',             type=str,   help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop',                      help='if set, crop input images as kitti benchmark images', action='store_true')

if sys.argv.__len__() > 1 and sys.argv[1][0] != '-':
	args = parser.parse_args(['@' + sys.argv[1]] + sys.argv[2:])
else:
	args = parser.parse_args()

def get_num_lines(file_path):
	f = open(file_path, 'r')
	lines = f.readlines()
	f.close()
	return len(lines)

def test(params):
	"""Test function."""
	model_file = os.path.join(args.checkpoint_path, args.model_name, 'final_model')
	
	reader = BtsReader(params)
	processor = BtsDataloader(params, do_kb_crop=args.do_kb_crop)

	if args.tfrecord_path is None or args.tfrecord_path == '':
		loader = reader.read_from_image_files(args.data_path, None, args.filenames_file, 'predict')
	else:
		loader = reader.read_from_tf_record(args.tfrecord_path, args.filenames_file, 'predict', args.tfrecord_shards)
	loader = processor.process_dataset(loader, 'predict')

	with tf.device('/cpu:0'):
		model, _ = bts_model(params, 'predict')
		model.compile(optimizer='adam', loss=None)
		print('Loading checkpoint at {}'.format(model_file))
		model.load_weights(model_file, by_name=False).expect_partial()
		print('Checkpoint successfully loaded')
		model_callbacks = [callbacks.ProgbarLogger(count_mode='steps')]

		model.summary()

		num_predict = get_num_lines(args.filenames_file)
		with open(args.filenames_file) as f:
			lines = f.readlines()
		print('Now executing inference on {} files with {}'.format(num_predict, args.checkpoint_path))
		
		print('Processing images..')
		pred_depths = model.predict(loader, verbose=1, callbacks=model_callbacks)

		save_name = 'result_{}'.format(args.model_name)
		print('Saving result pngs..')
		if not os.path.exists(os.path.dirname(save_name)):
			try:
				os.mkdir(save_name)
				os.mkdir(os.path.join(save_name, 'raw'))
				os.mkdir(os.path.join(save_name, 'cmap'))
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise

		for s in tqdm(range(num_predict)):
			if args.dataset == 'kitti':
				date_drive = lines[s].split('/')[1]
				filename_jpg = lines[s].split()[0].split('/')[-1]
				path_png = os.path.join(save_name, 'raw', '{}_{}'.format(date_drive, filename_jpg.replace('.jpg', '.png')))
				path_cmap_png = os.path.join(save_name, 'cmap', '{}_{}'.format(date_drive, filename_jpg.replace('.jpg', '.png')))
			elif args.dataset == 'kitti_benchmark':
				filename_jpg = lines[s].split()[0].split('/')[-1]
				path_png = os.path.join(save_name, 'raw', filename_jpg.replace('.jpg', '.png'))
				path_cmap_png = os.path.join(save_name, 'cmap', filename_jpg.replace('.jpg', '.png'))
			else:
				scene_name = lines[s].split()[0].split('/')[0]
				filename_jpg = lines[s].split()[0].split('/')[1]
				path_png = os.path.join(save_name, 'raw', '{}_{}'.format(scene_name, filename_jpg.replace('.jpg', '.png')))
				path_cmap_png = os.path.join(save_name, 'cmap', '{}_{}'.format(scene_name, filename_jpg.replace('.jpg', '.png')))

			pred_depth = pred_depths[s]

			if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
				pred_depth_scaled = pred_depth * 256.0
			else:
				pred_depth_scaled = pred_depth * 1000.0

			pred_depth_scaled *= 256 / args.max_depth
			pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
			cv2.imwrite(path_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

			if args.dataset == 'nyu':
				pred_depth_cropped = np.zeros((480, 640), dtype=np.float32) + 1
				pred_depth_cropped[10:-1 - 10, 10:-1 - 10] = pred_depth[10:-1 - 10, 10:-1 - 10]
				cv2.imwrite(path_cmap_png, pred_depth_cropped, [cv2.IMWRITE_PNG_COMPRESSION, 0])
			else:
				cv2.imwrite(path_cmap_png, pred_depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])

		return


def main():
	
	params = bts_parameters(
		encoder=args.encoder,
		height=args.input_height,
		width=args.input_width,
		batch_size=args.batch_size,
		dataset=args.dataset,
		max_depth=args.max_depth,
		num_threads=None,
		num_epochs=None,
		use_tpu=False)

	test(params)


if __name__ == '__main__':
	main()
