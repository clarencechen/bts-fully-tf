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
import argparse
import time
import datetime
import sys

from collections import namedtuple

bts_parameters = namedtuple('parameters', 'encoder, '
										  'height, width, '
										  'max_depth, '
										  'batch_size, '
										  'dataset, '
										  'num_devices, '
										  'num_threads, '
										  'num_epochs, '
										  'use_tpu, ')

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.keras import callbacks

from bts_dataloader import BtsReader, BtsDataloader
from bts import si_log_loss_wrapper, bts_model
from custom_callbacks import BatchLRScheduler, TensorboardPlusDepthImages
from adafactor import AdaFactor

def convert_arg_line_to_args(arg_line):
	for arg in arg_line.split():
		if not arg.strip():
			continue
		yield arg


parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts or densenet161_bts', default='densenet161_bts')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=False)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--tfrecord_path',             type=str,   help='path to the combined TFRecord dataset in zip format', required=False)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=5.0)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--pretrained_model',          type=str,   help='path to a pretrained model checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')

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

def train(strategy, params):

	# Define training constants derived from parameters
	training_samples = get_num_lines(args.filenames_file)
	if params.use_tpu:
		steps_per_epoch = np.floor(training_samples / params.batch_size).astype(np.int32)
	else:
		steps_per_epoch = np.ceil(training_samples / params.batch_size).astype(np.int32)
	total_steps = params.num_epochs * steps_per_epoch
	print("Total number of samples: {}".format(training_samples))
	print("Total number of steps: {}".format(total_steps))

	tensorboard_log_dir = os.path.join(args.log_directory, args.model_name, 'tensorboard')
	model_save_dir = os.path.join(args.log_directory, args.model_name, 'checkpoint')

	# Scale batch size and learning rate by number of distributed training cores
	start_lr = args.learning_rate * strategy.num_replicas_in_sync
	end_lr = args.end_learning_rate * strategy.num_replicas_in_sync if args.end_learning_rate > 0 else start_lr * 0.1
	poly_decay_fn = lambda step, lr: (start_lr -end_lr)*(1 - min(step, total_steps)/total_steps)**0.9 +end_lr

	if args.fix_first_conv_blocks or args.fix_first_conv_block:
		if args.fix_first_conv_blocks:
			print('Fixing first two conv blocks')
		else:
			print('Fixing first conv block')

	reader = BtsReader(params)
	processor = BtsDataloader(params, do_rotate=args.do_random_rotate, degree=args.degree, do_kb_crop=args.do_kb_crop)

	if args.tfrecord_path is None or args.tfrecord_path == '':
		loader = reader.read_from_image_files(args.data_path, args.gt_path, args.filenames_file, args.mode)
	else:
		loader = reader.read_from_tf_record(args.tfrecord_path, args.mode)

	loader = processor.process_dataset(loader, args.mode)

	with strategy.scope():
		model = bts_model(params, args.mode, fix_first=args.fix_first_conv_block, 
											 fix_first_two=args.fix_first_conv_blocks, 
											 pretrained_weights_path=args.pretrained_model)
		opt = AdaFactor(lr=start_lr, epsilon=1e-8)
		loss = si_log_loss_wrapper(params.dataset)

		# Load checkpoint if set
		initial_epoch = 0
		if args.checkpoint_path != '':
			checkpoint_file = os.path.join(args.checkpoint_path, 'checkpoint')
			print('Loading checkpoint at {}'.format(checkpoint_file))
			# model = tf.keras.models.load_model(checkpoint_file, custom_objects={'si_log_loss': loss}, compile=False)
			model.load_weights(checkpoint_file, by_name=False)
			if not args.retrain:
				# initial_epoch = (model.optmizer.iterations.value) // steps_per_epoch
				initial_epoch = 4
			print('Checkpoint successfully loaded')

		model.compile(optimizer=opt, loss=loss)

	model.summary()
	model_callbacks = [BatchLRScheduler(poly_decay_fn, steps_per_epoch, initial_epoch=initial_epoch, verbose=1),
					   TensorboardPlusDepthImages(params.height, params.width, params.max_depth, log_dir=tensorboard_log_dir),
					   callbacks.TerminateOnNaN(),
					   callbacks.ProgbarLogger(count_mode='steps'),
					   callbacks.ModelCheckpoint(model_save_dir,
							monitor='loss', mode='auto', verbose=1,
							save_best_only=True, save_weights_only=True)]

	model.fit(x=loader,
			  initial_epoch=initial_epoch,
			  epochs=params.num_epochs,
			  verbose=1,
			  callbacks=model_callbacks,
			  steps_per_epoch=steps_per_epoch)

	# model.save(model_save_dir, save_format='tf')
	model.save_weights(model_save_dir, save_format='tf')

	print('{} training finished at {}'.format(args.model_name, datetime.datetime.now()))


def main():
	if args.mode == 'train':
		# Find TPU cluster if available
		try:
			tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
			print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
		except ValueError:
			tpu = None

		# TPUStrategy for distributed training
		if tpu:
			tf.config.experimental_connect_to_cluster(tpu)
			tf.tpu.experimental.initialize_tpu_system(tpu)
			strategy = tf.distribute.experimental.TPUStrategy(tpu)
		else: # default strategy that works on CPU and single GPU
			strategy = tf.distribute.get_strategy()

		model_folder = os.path.join(args.log_directory, args.model_name)
		command = 'mkdir {}'.format(model_folder)
		os.system(command)

		params = bts_parameters(
			encoder=args.encoder,
			height=args.input_height,
			width=args.input_width,
			batch_size=args.batch_size * strategy.num_replicas_in_sync,
			dataset=args.dataset,
			max_depth=args.max_depth,
			num_devices=1 if tpu else args.num_gpus,
			num_threads=args.num_threads,
			num_epochs=args.num_epochs,
			use_tpu=False if tpu is None else True)

		train(strategy, params)
		
	elif args.mode == 'test':
		print('This script does not support testing. Use bts_test.py instead.')


if __name__ == '__main__':
	main()
