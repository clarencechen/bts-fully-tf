import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
try:
    from tensorflow.keras import initializations
except ImportError:
    from tensorflow.keras import initializers as initializations
import tensorflow.keras.backend as K

class LocalPlanarGuidance(layers.Conv2D):
	def __init__(self, height, width, upratio, **kwargs):
		super(LocalPlanarGuidance, self).__init__(filters=4, kernel_size=1, dilation_rate=1, padding='same', activation=None, **kwargs)
		self.full_height = height
		self.full_width = width
		self.upratio = upratio

	def build(self, input_shape):
		self.batch_size = input_shape[0]
		v, u = tf.meshgrid(
			np.linspace(0, self.full_width -1, self.full_width, dtype=np.float32), 
			np.linspace(0, self.full_height -1, self.full_height, dtype=np.float32))

		v = K.expand_dims(K.stack([v]*self.batch_size, axis=0), axis=-1)
		v = (v % self.upratio - (self.upratio - 1)/2) / K.reshape(float(self.upratio), (-1, 1, 1, 1)) # V2 Update

		u = K.expand_dims(K.stack([u]*self.batch_size, axis=0), axis=-1)
		u = (u % self.upratio - (self.upratio - 1)/2) / K.reshape(float(self.upratio), (-1, 1, 1, 1)) # V2 Update

		self.pixel_vector = K.l2_normalize(K.concatenate([u, v, K.ones_like(u)], axis=3), axis=3) # Normalize first

		return super(LocalPlanarGuidance, self).build(input_shape)

	def call(self, inputs):
		conv_out = super(LocalPlanarGuidance, self).call(inputs)

		raw_normal, raw_dist = K.tanh(conv_out[:, :, :, 0:3]), K.sigmoid(conv_out[:, :, :, 3:])
		plane_coeffs = K.concatenate([raw_normal, raw_dist], axis=3)

		plane = K.reshape(plane_coeffs, (self.batch_size, -1, 1, 4))
		plane = K.concatenate([plane]*self.upratio, axis=2)
		plane = K.reshape(plane, (self.batch_size, self.full_height//self.upratio, -1, 4))
		plane = K.concatenate([plane]*self.upratio, axis=2)
		plane = K.reshape(plane, (self.batch_size, self.full_height, self.full_width, 4))

		plane_normal, plane_dist = K.l2_normalize(plane[:, :, :, 0:3], axis=3), plane[:, :, :, 3:]
		denominator = K.sum(self.pixel_vector*plane_normal, axis=3, keepdims=True)
		return plane_dist / denominator # V2 Update (Normalize first)

	def get_config(self):
		base_config = super(LocalPlanarGuidance, self).get_config()
		config = {'upratio': self.upratio, 'full_height': self.full_height, 'full_width': self.full_width}
		return dict(list(base_config.items()) + list(config.items()))