from __future__ import absolute_import, division, print_function

import tensorflow as tf

def metrics_list_factory(args):
	
	@tf.function
	def pre_eval(y_true, y_pred):

		@tf.function
		def ground_truth_mask(y_true):
			valid_mask = tf.logical_and(y_true < args.max_depth_eval, y_true > args.min_depth_eval)
			eval_mask = tf.zeros_like(valid_mask)
			if args.garg_crop:
				eval_mask[:, int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width), ...].assign(1.0)
			elif args.eigen_crop:
				if args.dataset == 'kitti':
					eval_mask[:, int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width), ...].assign(1.0)
				else:
					eval_mask[:, 45:471, 41:601, ...].assign(1.0)
			return valid_mask

		mask = tf.logical_and(y_true < args.max_depth_eval, y_true > args.min_depth_eval)
		y_pred = tf.clip_by_value(y_pred, args.min_depth_eval, args.max_depth_eval)
		return tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask)

	@tf.function
	def d1(y_true, y_pred): 
		gt, pred = pre_eval(y_true, y_pred)
		return tf.reduce_mean(tf.cast(tf.maximum(gt/pred, pred/gt) < 1.25, dtype=tf.float32))
	@tf.function
	def d2(y_true, y_pred):
		gt, pred = pre_eval(y_true, y_pred)
		return tf.reduce_mean(tf.cast(tf.maximum(gt/pred, pred/gt) < 1.25 ** 2, dtype=tf.float32))
	@tf.function
	def d3(y_true, y_pred):
		gt, pred = pre_eval(y_true, y_pred)
		return tf.reduce_mean(tf.cast(tf.maximum(gt/pred, pred/gt) < 1.25 ** 3, dtype=tf.float32))

	@tf.function
	def rmse(y_true, y_pred):
		gt, pred = pre_eval(y_true, y_pred)
		return tf.sqrt(tf.reduce_mean((gt - pred)**2))
	@tf.function
	def rmse_log(y_true, y_pred):
		gt, pred = pre_eval(y_true, y_pred)
		d = tf.math.log(gt) -tf.math.log(pred)
		return tf.sqrt(tf.reduce_mean(d**2))

	@tf.function
	def abs_rel(y_true, y_pred):
		gt, pred = pre_eval(y_true, y_pred)
		return tf.reduce_mean(tf.abs(gt - pred)/gt)
	@tf.function
	def sq_rel(y_true, y_pred):
		gt, pred = pre_eval(y_true, y_pred)
		return tf.reduce_mean(((gt - pred)**2)/gt)

	@tf.function
	def silog(y_true, y_pred):
		gt, pred = pre_eval(y_true, y_pred)
		d = tf.math.log(gt) -tf.math.log(pred)
		return tf.sqrt(tf.reduce_mean(d**2) - tf.reduce_mean(d)**2) * 100

	@tf.function
	def log10(y_true, y_pred):
		gt, pred = pre_eval(y_true, y_pred)
		d = tf.math.log(gt) - tf.math.log(pred)
		return tf.reduce_mean(tf.abs(d))/tf.math.log(10.0)

	return [silog, abs_rel, log10, rmse, sq_rel, rmse_log, d1, d2, d3]

