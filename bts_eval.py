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
import cv2
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
from tensorflow.keras import callbacks

from bts_dataloader import BtsReader, BtsDataloader
from bts import si_log_loss_wrapper, bts_model
from custom_eval_metrics import metrics_list_factory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def convert_arg_line_to_args(arg_line):
	for arg in arg_line.split():
		if not arg.strip():
			continue
		yield arg


parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',          type=str,   help='model name', default='bts_v0_0_1')
parser.add_argument('--encoder',             type=str,   help='type of encoder, desenet121_bts or densenet161_bts', default='densenet161_bts')
parser.add_argument('--data_path',           type=str,   help='path to the data', required=False)
parser.add_argument('--gt_path',             type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--tfrecord_path',       type=str,   help='path to the combined TFRecord dataset in zip format', required=False)
parser.add_argument('--tfrecord_shards',     type=int,   help='number of shards of the combined TFRecord dataset to read', default=1)
parser.add_argument('--filenames_file',      type=str,   help='path to the filenames text file', required=True)

parser.add_argument('--input_height',        type=int,   help='input height', default=480)
parser.add_argument('--input_width',         type=int,   help='input width', default=640)
parser.add_argument('--max_depth',           type=float, help='maximum depth in estimation',        default=80)

parser.add_argument('--output_directory',    type=str,   help='output directory for summary, if empty outputs to checkpoint folder', default='')
parser.add_argument('--checkpoint_path',     type=str,   help='path to a specific checkpoint to load', default='', required=True)
parser.add_argument('--dataset',             type=str,   help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',     action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',     action='store_true')

parser.add_argument('--min_depth_eval',      type=float, help='minimum depth for evaluation',        default=1e-3)
parser.add_argument('--max_depth_eval',      type=float, help='maximum depth for evaluation',        default=80)
parser.add_argument('--do_kb_crop',                      help='if set, crop input images as kitti benchmark images', action='store_true')

parser.add_argument('--batch_size',          type=int,   help='batch size per training replica', default=1)
parser.add_argument('--num_gpus',            type=int,   help='number of GPUs to use for evaluation', default=1)
parser.add_argument('--num_threads',         type=int,   help='number of threads to use for data loading', default=8)

if sys.argv.__len__() > 1 and sys.argv[1][0] != '-':
	args = parser.parse_args(['@' + sys.argv[1]] + sys.argv[2:])
else:
	args = parser.parse_args()



def get_num_lines(file_path):
	f = open(file_path, 'r')
	lines = f.readlines()
	f.close()
	return len(lines)


def test(strategy, params):
	checkpoint_file = os.path.join(args.checkpoint_path, args.model_name, 'checkpoint')
	if not args.output_directory:
		tensorboard_log_dir = os.path.join(args.output_directory, args.model_name, 'tensorboard')
	else:
		tensorboard_log_dir = os.path.join(args.checkpoint_path, args.model_name, 'tensorboard')

	reader = BtsReader(params)
	processor = BtsDataloader(params, do_kb_crop=args.do_kb_crop)

	if args.tfrecord_path is None or args.tfrecord_path == '':
		loader = reader.read_from_image_files(args.data_path, args.gt_path, args.filenames_file, 'test')
	else:
		loader = reader.read_from_tf_record(args.tfrecord_path, args.filenames_file, 'test', args.tfrecord_shards)

	loader = processor.process_dataset(loader, 'test')

	with strategy.scope():
		model = bts_model(params, 'test')
		loss = si_log_loss_wrapper(params.dataset)
		model.compile(optimizer='adam', loss=loss, metrics=metrics_list_factory(args))
		# Load checkpoint if set
		print('Loading checkpoint at {}'.format(checkpoint_file))
		model.load_weights(checkpoint_file, by_name=False).expect_partial()
		print('Checkpoint successfully loaded')

	model_callbacks = [callbacks.TensorBoard(log_dir=tensorboard_log_dir, write_graph=False),
					   callbacks.ProgbarLogger(count_mode='steps')]

	model.summary()

	num_test_samples = get_num_lines(args.filenames_file)
	with open(args.filenames_file) as f:
		lines = f.readlines()
	print('Now testing {} images.'.format(num_test_samples))

	metrics = model.evaluate(loader, verbose=1, callbacks=model_callbacks)
	print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
	print("{0[1]:7.4f}, {0[2]:7.4f}, {0[3]:7.3f}, {0[4]:7.3f}, {0[5]:7.3f}, {0[6]:7.3f}, {0[7]:7.3f}, {0[8]:7.3f}, {0[9]:7.3f}".format(metrics))


def main():
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
	# MirroredStrategy for distributed training on multiple GPUs
	elif args.num_gpus > 1:
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if len(gpus) < args.num_gpus:
			raise ValueError('Insufficient resources for distributed training: {} GPUs available out of {} GPUs requested.'.format(len(gpus), args.num_gpus))
		gpu_name_list = [gpu.name for gpu in gpus]
		strategy = tf.distribute.MirroredStrategy(devices=gpu_name_list[:num_gpus])
	# Default strategy that works on CPU and single GPU
	else:
		strategy = tf.distribute.get_strategy()

	params = bts_parameters(
		encoder=args.encoder,
		height=args.input_height,
		width=args.input_width,
		batch_size=args.batch_size * strategy.num_replicas_in_sync,
		dataset=args.dataset,
		max_depth=args.max_depth,
		num_threads=args.num_threads,
		num_epochs=None,
		use_tpu=False if tpu is None else True)

	test(strategy, params)


if __name__ == '__main__':
	main()
