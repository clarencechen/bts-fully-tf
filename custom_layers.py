from tf.keras import layers
try:
    from tf.keras import initializations
except ImportError:
    from tf.keras import initializers as initializations
import tf.keras.backend as K

class UpSample(layers.Layer):
	def __init__(self, ratio, method='NN', **kwargs):
		super(UpSample, self).__init__(**kwargs)
		self.ratio = ratio
		self.method = method

	def build(self, input_shape):
		self.h = input_shape[1]
		self.w = input_shape[2]
		return super(UpSample, self).build(input_shape)

	def call(self, inputs):
		if self.method == 'bilinear': 
			return tf.compat.v1.image.resize_bilinear(inputs, [self.h * self.ratio, self.w * self.ratio], align_corners=True)
		else:
			return tf.compat.v1.image.resize_nearest_neighbor(inputs, [self.h * self.ratio, self.w * self.ratio], align_corners=True)

	def get_config(self):
		base_config = super(UpSample, self).get_config()
		config = {'ratio': self.ratio, 'method': self.method}
		return dict(list(base_config.items()) + list(config.items()))

class DownSample(layers.Layer):
	def __init__(self, ratio, method='NN', **kwargs):
		super(DownSample, self).__init__(**kwargs)
		self.ratio = ratio
		self.method = method

	def build(self, input_shape):
		self.h = tf.cast(input_shape[1] / self.ratio, tf.int32)
		self.w = tf.cast(input_shape[2] / self.ratio, tf.int32)
		return super(DownSample, self).build(input_shape)
	
	def call(self, inputs):
		if self.method == 'bilinear': 
			return tf.compat.v1.image.resize_bilinear(inputs, [self.h, self.w], align_corners=True)
		else:
			return tf.compat.v1.image.resize_nearest_neighbor(inputs, [self.h, self.w], align_corners=True)

	def get_config(self):
		base_config = super(DownSample, self).get_config()
		config = {'ratio': self.ratio, 'method': self.method}
		return dict(list(base_config.items()) + list(config.items()))

class LocalPlanarGuidance(layers.Conv2D):
	def __init__(self, height, width, upratio, **kwargs):
		super(LocalPlanarGuidance, self).__init__(filters=4, kernel_size=1, dilation_rate=1, padding='same', activation_fn=None, **kwargs)
		self.full_height = height
		self.full_width = width
		self.upratio = upratio

	def build(self, input_shape):
		self.batch_size = input_shape[0]
		v, u = tf.meshgrid(
			np.linspace(0, self.full_width -1, self.full_width, dtype=np.float32), 
			np.linspace(0, self.full_height -1, self.full_height, dtype=np.float32))

		v = K.expand_dims(K.stack([v]*self.batch_size, axis=0), axis=-1)
		v = (v % self.upratio - (self.upratio - 1)/2) / K.reshape(self.upratio, (-1, 1, 1, 1)) # V2 Update

		u = K.expand_dims(K.stack([u]*self.batch_size, axis=0), axis=-1)
		u = (u % self.upratio - (self.upratio - 1)/2) / K.reshape(self.upratio, (-1, 1, 1, 1)) # V2 Update

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
		config = {'upratio': self.ratio, 'full_height': self.full_height, 'full_width': self.full_width}
		return dict(list(base_config.items()) + list(config.items()))