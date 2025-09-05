import os
import numpy as np
import tensorflow as tf

IMG_SIZE = (256, 256)
N_COEFFS = 5			# 4 degree polynomial
TRAIN_VAL_RATIO = 0.8	# train / val ratio

def load_rgbd(root_dataset_path, index):
	rgbd = np.load(root_dataset_path + f"/{index}_rgbd.npy")
	assert rgbd.shape == (IMG_SIZE[0], IMG_SIZE[1], 4), f"Invalid shape for rgbd: {rgbd.shape}"
	return rgbd

def load_coeffs(root_dataset_path, index):
	coeffs = np.load(root_dataset_path + f"/{index}_coeffs.npy")
	assert coeffs.shape == (N_COEFFS,), f"Invalid shape for coeffs: {coeffs.shape}"
	return coeffs


def load_dataset(root_dataset_path, batch_size=32):
	"""
	Returns:
		train_ds: tf.data.Dataset
		val_ds: tf.data.Dataset
		coeff_scaling_factors: np.ndarray
		raw_relative_depth_samples: np.ndarray
	"""

	ds_rgbd_image_count = len([
		file for file in os.listdir(root_dataset_path)
		if file.endswith("_rgbd.npy")
	])

	train_count = int(ds_rgbd_image_count * TRAIN_VAL_RATIO)

	train_indices = tf.data.Dataset.from_tensor_slices(np.arange(train_count))
	val_indices   = tf.data.Dataset.from_tensor_slices(np.arange(train_count, ds_rgbd_image_count))


	# find a and b for: scaled_coeff(coeff) = a * coeff + b,
	# such that scaled_coeff fits in [-1,1] in the majority of cases
	training_coeffs = np.zeros((train_count, N_COEFFS), dtype=np.float32)
	for i in train_indices:
		coeffs = load_coeffs(root_dataset_path, i)
		training_coeffs[i] = coeffs

	coeff_scaling_factors = np.zeros((N_COEFFS, 2), dtype=np.float32)
	for i in range(N_COEFFS):
		low = np.percentile(training_coeffs[:, i], 0.5)
		high = np.percentile(training_coeffs[:, i], 99.5)

		a = 2.0 / (high - low)
		b = -1 - a * low
		coeff_scaling_factors[i, 0] = a
		coeff_scaling_factors[i, 1] = b

	coeffs_scaling_factor_a = coeff_scaling_factors[:, 0]
	coeffs_scaling_factor_b = coeff_scaling_factors[:, 1]

	raw_relative_depth_samples = np.load(os.path.join(root_dataset_path, 'raw_relative_depth_samples.npy'))

	def map_fn(idx):
		rgbd = tf.numpy_function(
			func=lambda i: load_rgbd(root_dataset_path, int(i)),
			inp=[idx],
			Tout=tf.float32
		)
		rgbd.set_shape((*IMG_SIZE, 4))

		def scale_coeffs(coeffs):
			scaled_coeffs = coeffs_scaling_factor_a * coeffs + coeffs_scaling_factor_b
			return tf.clip_by_value(scaled_coeffs, -1, 1)

		coeffs = tf.numpy_function(
			func=lambda i: scale_coeffs(load_coeffs(root_dataset_path, int(i))),
			inp=[idx],
			Tout=tf.float32
		)
		coeffs.set_shape((N_COEFFS,))

		return rgbd, coeffs

	def augment_dataset_with_horizontal_flip(ds):
		flipped = ds.map(lambda rgbd, coeffs: (tf.image.flip_left_right(rgbd), coeffs))
		return ds.concatenate(flipped)


	train_ds = (
		train_indices
			.shuffle(buffer_size=train_count)
			.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
			.apply(augment_dataset_with_horizontal_flip)
			.batch(batch_size)
			.prefetch(tf.data.AUTOTUNE)
	)

	val_ds = (
		val_indices
			.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
			.apply(augment_dataset_with_horizontal_flip)
			.batch(batch_size)
			.prefetch(tf.data.AUTOTUNE)
	)

	return train_ds, val_ds, coeff_scaling_factors, raw_relative_depth_samples