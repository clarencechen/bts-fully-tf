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
import numpy as np
import cv2
import sys

from bts_dataloader import *

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
parser.add_argument('--data_path',           type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',             type=str,   help='path to the groundtruth data', required=False)
parser.add_argument('--filenames_file',      type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',        type=int,   help='input height', default=480)
parser.add_argument('--input_width',         type=int,   help='input width', default=640)
parser.add_argument('--max_depth',           type=float, help='maximum depth in estimation',        default=80)
parser.add_argument('--output_directory',    type=str,   help='output directory for summary, if empty outputs to checkpoint folder', default='')
parser.add_argument('--checkpoint_path',     type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset',             type=str,   help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',     action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',     action='store_true')

parser.add_argument('--min_depth_eval',      type=float, help='minimum depth for evaluation',        default=1e-3)
parser.add_argument('--max_depth_eval',      type=float, help='maximum depth for evaluation',        default=80)
parser.add_argument('--do_kb_crop',                      help='if set, crop input images as kitti benchmark images', action='store_true')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
	global gt_depths, is_missing, missing_ids
	gt_depths = []
	is_missing = []
	missing_ids = set()

	checkpoint_file = os.path.join(args.checkpoint_path, args.model_name, 'checkpoint')
	tensorboard_log_dir = os.path.join(args.output_directory, args.model_name, 'tensorboard')

	reader = BtsReader(params)
	processor = BtsDataloader(params, do_kb_crop=args.do_kb_crop)

	if args.tfrecord_path is None or args.tfrecord_path == '':
		loader = reader.read_from_image_files(args.data_path, args.gt_path, args.filenames_file, 'test')
	else:
		loader = reader.read_from_tf_record(args.tfrecord_path, 'test')

	loader = processor.process_dataset(loader, 'test')

	model = bts_model(params, 'test')
	model.compile(optimizer='adam', metrics=metrics_list_factory(args))
	# Load checkpoint if set
	print('Loading checkpoint at {}'.format(checkpoint_file))
	model.load_weights(checkpoint_file, by_name=False).expect_partial()
	print('Checkpoint successfully loaded')

	model_callbacks = [callbacks.TensorBoard(log_dir=tensorboard_log_dir, write_graph=False),
					   callbacks.ProgbarLogger(count_mode='steps')]

	model.summary()

	with tf.device('/cpu:0'):
		num_test_samples = get_num_lines(args.filenames_file)
		with open(args.filenames_file) as f:
			lines = f.readlines()
		print('Now testing {} images.'.format(num_test_samples))

		start_time = time.time()
		metrics = model.evaluate(loader, verbose=1, callbacks=model_callbacks)
		elapsed_time = time.time() - start_time
		print('Evaluation finished. Elapsed time: {}'.format(str(elapsed_time)))
		print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
		print("{0[0]:7.4f}, {0[1]:7.4f}, {0[2]:7.3f}, {0[3]:7.3f}, {0[4]:7.3f}, {0[5]:7.3f}, {0[6]:7.3f}, {0[7]:7.3f}, {0[8]:7.3f}".format(metrics))


def main():
	
	params = bts_parameters(
		encoder=args.encoder,
		height=args.input_height,
		width=args.input_width,
		batch_size=1,
		dataset=args.dataset,
		max_depth=args.max_depth,
		num_devices=1,
		num_threads=args.num_threads,
		num_epochs=None,
		use_tpu=False)

	test(params)
>>>>>>> daf7cac... load_weights also includes optimizer iterations


if __name__ == '__main__':
    tf.compat.v1.app.run()



