import os
import math
import numpy as np
import tensorflow as tf

IMG_SIZE = (256, 256)
N_COEFFS = 5			# 4 degree polynomial
TRAIN_VAL_RATIO = 0.8	# train / val ratio

AUGMENTATION_ROTATION_ANGLES = [-15.0, 15.0]

SAMPLE_REL_ABS_PAIRS_COUNT = 1024

def sample_rel_abs_pairs(rel_abs_pairs):
    rel_abs_pairs = np.array(rel_abs_pairs, dtype=np.float32)
    n = rel_abs_pairs.shape[0]

    if n >= SAMPLE_REL_ABS_PAIRS_COUNT:
        indices = np.random.choice(n, SAMPLE_REL_ABS_PAIRS_COUNT, replace=False)
    else:
        indices = np.random.choice(n, SAMPLE_REL_ABS_PAIRS_COUNT, replace=True)

    return rel_abs_pairs[indices]

def rotate_and_crop(image, angle_degrees):
	"""
	Rotate `image` by angle_degrees around center, crop to the largest safe
	rectangle (so no padding is visible), then resize back to original HxW.
	Works with tf.data (no .numpy()) and uses tf.raw_ops.ImageProjectiveTransformV3.
	"""
	# angle in radians (python float -> TF scalar)
	angle = tf.cast(angle_degrees * math.pi / 180.0, tf.float32)

	# image shape (symbolic)
	h = tf.shape(image)[0]
	w = tf.shape(image)[1]

	# rotation matrix entries
	cos_a = tf.math.cos(angle)
	sin_a = tf.math.sin(angle)

	# Build 8-element projective transform vector for affine rotation:
	# a0 a1 a2 a3 a4 a5 a6 a7  where a6=a7=0 for affine
	# For rotation: [ cos -sin 0, sin cos 0, 0, 0 ]
	transform = tf.stack([cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0])
	transform = tf.reshape(transform, [1, 8])                 # shape [1,8]
	transform = tf.cast(transform, tf.float32)

	# Apply projective transform (expects batch images and transforms shape [N,8] or [1,8])
	rotated = tf.raw_ops.ImageProjectiveTransformV3(
		images=tf.expand_dims(image, 0),   # add batch dim -> [1, H, W, C]
		transforms=transform,             # [1,8]
		output_shape=[h, w],              # output H,W (symbolic allowed)
		interpolation="BILINEAR",
		fill_mode="REFLECT",              # fill mode (ignored after crop)
		fill_value=0.0
	)
	rotated = tf.squeeze(rotated, axis=0)  # back to [H, W, C]

	# --- Safe center crop calculation (all TF ops) ---
	abs_cos = tf.abs(cos_a)
	abs_sin = tf.abs(sin_a)

	# bound dims (floats -> ints)
	bound_w = tf.cast(tf.cast(w, tf.float32) * abs_cos + tf.cast(h, tf.float32) * abs_sin, tf.int32)
	bound_h = tf.cast(tf.cast(h, tf.float32) * abs_cos + tf.cast(w, tf.float32) * abs_sin, tf.int32)

	# crop size proportional to original
	crop_w = tf.cast(tf.cast(w, tf.float32) * (tf.cast(w, tf.float32) / tf.cast(bound_w, tf.float32)), tf.int32)
	crop_h = tf.cast(tf.cast(h, tf.float32) * (tf.cast(h, tf.float32) / tf.cast(bound_h, tf.float32)), tf.int32)

	# offsets must be >=0
	offset_x = tf.maximum((w - crop_w) // 2, 0)
	offset_y = tf.maximum((h - crop_h) // 2, 0)

	# crop and resize back to original size
	cropped = tf.image.crop_to_bounding_box(rotated, offset_y, offset_x, crop_h, crop_w)
	resized = tf.image.resize(cropped, (h, w), method="bilinear")

	# keep original dtype (e.g. float32)
	resized = tf.cast(resized, image.dtype)
	return resized

def augment_brightness(rgbd, brightness):
	"""Does not change the associated depth, only the brightness of the rgb image"""
	rgb = rgbd[..., :3]
	rgb = ((brightness * 2.0) * (rgb + 1.0) / 2.0) - 1.0
	rgb = tf.clip_by_value(rgb, -1.0, 1.0)
	depth = rgbd[..., 3:]
	return tf.concat([rgb, depth], axis=-1)



def load_rgbd(root_dataset_path, index):
	rgbd = np.load(root_dataset_path + f"/{index}_rgbd.npy")
	assert rgbd.shape == (IMG_SIZE[0], IMG_SIZE[1], 4), f"Invalid shape for rgbd: {rgbd.shape}"
	return rgbd

def load_coeffs(root_dataset_path, index):
	coeffs = np.load(root_dataset_path + f"/{index}_coeffs.npy")
	assert coeffs.shape == (N_COEFFS,), f"Invalid shape for coeffs: {coeffs.shape}"
	return coeffs

def load_rel_abs_pairs(root_dataset_path, index):
	rel_abs_pairs = np.load(root_dataset_path + f"/{index}_rel_abs_pairs.npy")
	rel_abs_pairs = rel_abs_pairs.reshape(-1, 2)
	return rel_abs_pairs


def load_dataset(root_dataset_path, batch_size=4):
	"""
	Returns:
		train_ds: tf.data.Dataset
		val_ds: tf.data.Dataset
		coeff_scaling_factors: np.ndarray
	"""

	ds_rgbd_image_count = 100#len([
		#file for file in os.listdir(root_dataset_path)
		#if file.endswith("_rgbd.npy")
	#])

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

	def map_fn(idx):
		rgbd = tf.numpy_function(
			func=lambda i: load_rgbd(root_dataset_path, int(i)),
			inp=[idx],
			Tout=tf.float32
		)
		rgbd.set_shape((*IMG_SIZE, 4))

		rel_abs_pairs = tf.numpy_function(
			func=lambda i: sample_rel_abs_pairs(load_rel_abs_pairs(root_dataset_path, int(i))),
			inp=[idx],
			Tout=tf.float32
		)
		rel_abs_pairs.set_shape((SAMPLE_REL_ABS_PAIRS_COUNT, 2))

		return rgbd, rel_abs_pairs

	def augment_dataset(ds):
		# 1. Horizontal flip
		flipped = ds.map(lambda rgbd, rel_abs_pairs: (tf.image.flip_left_right(rgbd), rel_abs_pairs))
		original_and_flipped = ds.concatenate(flipped)

		# 2. Rotations
		all_datasets_augmented = [original_and_flipped]  # start with original + flipped

		for angle in AUGMENTATION_ROTATION_ANGLES:
			# Randomly augment brightness (20% chance)
			if tf.random.uniform(()) > 0.8:
				brightness_percentage = tf.random.uniform((), 0.7, 1.3)
				rotated = original_and_flipped.map(lambda rgbd, rel_abs_pairs: (augment_brightness(rotate_and_crop(rgbd, angle), brightness_percentage), rel_abs_pairs))
			else:
				rotated = original_and_flipped.map(lambda rgbd, rel_abs_pairs: (rotate_and_crop(rgbd, angle), rel_abs_pairs))
			all_datasets_augmented.append(rotated)


		augmented = all_datasets_augmented[0]
		for ds_part in all_datasets_augmented[1:]:
			augmented = augmented.concatenate(ds_part)

		return augmented

	train_ds = (
		train_indices
			.shuffle(buffer_size=train_count)
			.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
			.apply(augment_dataset)
			.batch(batch_size)
			.prefetch(tf.data.AUTOTUNE)
	)

	val_ds = (
		val_indices
			.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
			.batch(batch_size)
			.prefetch(tf.data.AUTOTUNE)
	)

	return train_ds, val_ds, coeff_scaling_factors
