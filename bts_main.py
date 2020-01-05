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
										  'num_gpus, '
										  'num_threads, '
										  'num_epochs, ')

from tensorflow.python import pywrap_tensorflow
from tensorflow.keras import callbacks

from bts_dataloader import *
from bts import si_log_loss_wrapper, bts_model
from custom_callbacks import BatchLRScheduler, TensorboardPlusDepthImages

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
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=False)
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

if args.mode == 'train' and not args.checkpoint_path:
	from bts import *

elif args.mode == 'train' and args.checkpoint_path:
	model_dir = os.path.dirname(args.checkpoint_path)
	model_name = os.path.basename(model_dir)
	import sys
	sys.path.append(model_dir)
	for key, val in vars(__import__(model_name)).items():
		if key.startswith('__') and key.endswith('__'):
			continue
		vars()[key] = val


def get_num_lines(file_path):
	f = open(file_path, 'r')
	lines = f.readlines()
	f.close()
	return len(lines)

def train(params):
	training_samples = get_num_lines(args.filenames_file)

	steps_per_epoch = np.ceil(training_samples / params.batch_size).astype(np.int32)
	total_steps = params.num_epochs * steps_per_epoch

	start_lr = args.learning_rate
	end_lr = args.end_learning_rate if args.end_learning_rate > 0 else start_lr * 0.1
	poly_decay_fn = lambda step, lr: (start_lr -end_lr)*(1 - min(step, total_steps)/total_steps)**0.9 +end_lr

	print("Total number of samples: {}".format(training_samples))
	print("Total number of steps: {}".format(total_steps))
	if args.fix_first_conv_blocks or args.fix_first_conv_block:
		if args.fix_first_conv_blocks:
			print('Fixing first two conv blocks')
		else:
			print('Fixing first conv block')

	dataloader = BtsDataloader(args.data_path, args.gt_path, args.filenames_file, params, args.mode,
							   do_rotate=args.do_random_rotate, degree=args.degree,
							   do_kb_crop=args.do_kb_crop)
	
	tensorboard_log_dir = '{}/{}'.format(args.log_directory, args.model_name)
	
	model_save_dir = '{}/{}/model'.format(args.log_directory, args.model_name)
	
	model = bts_model(params, args.mode, start_lr, fix_first=args.fix_first_conv_block, 
												   fix_first_two=args.fix_first_conv_blocks, 
												   pretrained_weights_path=args.pretrained_model)
	opt = tf.keras.optimizers.Adam(lr=start_lr, epsilon=1e-8)
	loss = si_log_loss_wrapper(params.dataset)
	model.compile(optimizer=opt, loss=loss)
	model.summary()

	# Load checkpoint if set
	if args.checkpoint_path != '':
		print('Loading checkpoint at {}'.format(args.checkpoint_path))
		model = tf.keras.models.load_model(args.checkpoint_path, custom_objects={'si_log_loss': loss}, compile=False)
		model.compile(optimizer=opt, loss=loss)
		print('Checkpoint successfully loaded')
		if args.retrain:
			initial_epoch = 0
		else:
			initial_epoch = 6
	else:
		initial_epoch = 0

	model_callbacks = [BatchLRScheduler(poly_decay_fn, steps_per_epoch, initial_epoch=initial_epoch, verbose=1),
					   callbacks.TerminateOnNaN(),
					   callbacks.TensorBoard(log_dir=tensorboard_log_dir),
					   callbacks.ProgbarLogger(count_mode='steps'),
					   callbacks.ModelCheckpoint(model_save_dir, monitor='loss', save_best_only=True, mode='auto', save_freq=500*params.batch_size)]

	model.fit(x=dataloader.loader,
			  initial_epoch=initial_epoch,
			  epochs=params.num_epochs,
			  verbose=1,
			  callbacks=model_callbacks,
			  steps_per_epoch=steps_per_epoch)

	model.save(model_save_dir, save_format='tf')
	print('{} training finished at {}'.format(args.model_name), datetime.datetime.now())


def main():
	
	params = bts_parameters(
		encoder=args.encoder,
		height=args.input_height,
		width=args.input_width,
		batch_size=args.batch_size,
		dataset=args.dataset,
		max_depth=args.max_depth,
		num_gpus=args.num_gpus,
		num_threads=args.num_threads,
		num_epochs=args.num_epochs)

	if args.mode == 'train':
		model_filename = args.model_name + '.py'
		command = 'mkdir {}/{}'.format(args.log_directory, args.model_name)
		os.system(command)

		command = 'cp {} {}/{}/{}'.format(sys.argv[1], args.log_directory, args.model_name, sys.argv[1])
		os.system(command)

		if args.checkpoint_path == '':
			command = 'cp bts.py {}/{}/{}'.format(args.log_directory, args.model_name, model_filename)
			os.system(command)
		else:
			loaded_model_dir = os.path.dirname(args.checkpoint_path)
			loaded_model_filename = os.path.basename(loaded_model_dir) + '.py'

			command = 'cp {}/{} {}/{}/{}'.format(loaded_model_dir, loaded_model_filename, args.log_directory, args.model_name, model_filename)
			os.system(command)

		train(params)
		
	elif args.mode == 'test':
		print('This script does not support testing. Use bts_test.py instead.')


if __name__ == '__main__':
	main()
