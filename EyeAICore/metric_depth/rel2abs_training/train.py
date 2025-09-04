import sys
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from dataset import load_dataset, IMG_SIZE, N_COEFFS

LEARNING_RATE = 1e-4
EPOCHS = 25

# how many sample points between 0 and 1500 should be used for custom loss function
LOSS_SAMPLE_COUNT = 100
# sample values (raw relative depth)
RELATIVE_DEPTH_SAMPLES = tf.constant(tf.cast(tf.linspace(0, 1500, LOSS_SAMPLE_COUNT), tf.float32), dtype=tf.float32)

def sample_true_and_pred(scaled_y_true, scaled_y_pred, coeff_scaling_factors):
	"""
	Calculate sample metric output values based on the predicted and true coefficients

	Returns: y_true_eval, y_pred_eval
	"""

	a = tf.constant(coeff_scaling_factors[:,0], dtype=tf.float32)
	b = tf.constant(coeff_scaling_factors[:,1], dtype=tf.float32)

	unscaled_y_true= (scaled_y_true - b) / a
	unscaled_y_pred= (scaled_y_pred - b) / a

	# unscaled_y_true, unscaled_y_pred: [batch, N_COEFFS]
	powers = tf.range(tf.shape(unscaled_y_true)[1], dtype=tf.float32)  # [N_COEFFS]

	# [batch, m, N_COEFFS]
	x_powers = tf.pow(tf.expand_dims(RELATIVE_DEPTH_SAMPLES, -1), powers)

	# Evaluate polynomials
	y_true_eval = tf.matmul(x_powers, unscaled_y_true, transpose_b=True)  # shape [m, batch]
	y_pred_eval = tf.matmul(x_powers, unscaled_y_pred, transpose_b=True)  # shape [m, batch]

	y_true_eval = tf.transpose(y_true_eval)  # [batch, m]
	y_pred_eval = tf.transpose(y_pred_eval)

	return y_true_eval, y_pred_eval


def custom_loss_fn(coeff_scaling_factors):
	def custom_loss(y_true, y_pred):
		"""
		Custom loss function that calculates the loss of sample points generated using the coeffs predicted,
		instead of only using the loss of comparing the coeffs directly.
		This improves training, as there are coeffs that have way more impact on the actual result,
		such that treating every coeff equally is not optimal.
		"""

		y_true_eval, y_pred_eval = sample_true_and_pred(y_true, y_pred, coeff_scaling_factors)

		return tf.reduce_mean(tf.square(y_true_eval - y_pred_eval))  # MSE

	return custom_loss

def custom_mae_fn(coeff_scaling_factors):
	def custom_mae(y_true, y_pred):
		"""
		Just for displaying the same kind of metric as the actual training (which uses custom_loss)
		"""

		y_true_eval, y_pred_eval = sample_true_and_pred(y_true, y_pred, coeff_scaling_factors)

		return tf.reduce_mean(tf.abs(y_true_eval - y_pred_eval))	# MAE

	return custom_mae


def build_rel2abs_scaled_model():
	"""
	Creates a rel2abs model that fuses RGB and Depth features
	"""

	# Inputs
	inputs = layers.Input(shape=(*IMG_SIZE, 4), name="rgbd_input")
	rgb = layers.Lambda(lambda x: x[..., :3])(inputs)   # first 3 channels
	depth = layers.Lambda(lambda x: x[..., 3:4])(inputs) # last channel

	# --- RGB branch (pretrained MobileNetV3Large) ---
	base_model = tf.keras.applications.MobileNetV3Large(
		input_shape=(*IMG_SIZE, 3),
		include_top=False,
		weights="imagenet"
	)
	base_model.trainable = False
	rgb_features = base_model(rgb)
	rgb_features = layers.GlobalAveragePooling2D()(rgb_features)

	# --- Depth branch ---
	x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(depth)
	x = layers.MaxPooling2D((2, 2))(x)
	x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
	x = layers.MaxPooling2D((2, 2))(x)
	x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
	x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(256, activation="relu")(x)
	x = layers.Dropout(0.3)(x)
	depth_features = layers.Dense(128, activation="relu")(x)

	# --- Fusion ---
	fused = layers.Concatenate()([rgb_features, depth_features])
	fused = layers.Dense(256, activation="relu")(fused)
	fused = layers.Dropout(0.3)(fused)
	fused = layers.Dense(128, activation="relu")(fused)

	# Output
	scaled_coeffs_output = layers.Dense(N_COEFFS, activation=None, name="scaled_coeffs_output")(fused)

	model = tf.keras.models.Model(inputs=inputs, outputs=scaled_coeffs_output)
	return model

def unscale_rel2abs_model(scaled_model, coeff_scaling_factors):
	a = tf.constant(coeff_scaling_factors[:,0], dtype=tf.float32)
	b = tf.constant(coeff_scaling_factors[:,1], dtype=tf.float32)

	def unscale_fn(scaled_coeff):
		return (scaled_coeff - b) / a

	input = scaled_model.input
	scaled_output = scaled_model.output
	output = tf.keras.layers.Lambda(unscale_fn, name="coeffs_output", output_shape=(N_COEFFS,))(scaled_output)

	unscaled_rel2abs_model = tf.keras.Model(inputs=input, outputs=output)
	return unscaled_rel2abs_model

def export_as_tflite_model(keras_model, tflite_filepath):
	converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()

	with open(tflite_filepath, "wb") as f:
		f.write(tflite_model)

	print(f"Exported {tflite_filepath}")

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python train.py <path_to_prepared_dataset>")
		sys.exit(1)

	dataset_root_path = sys.argv[1]

	train_ds, val_ds, coeff_scaling_factors = load_dataset(dataset_root_path)

	scaled_rel2abs_model = build_rel2abs_scaled_model()
	scaled_rel2abs_model.compile(
		optimizer=optimizers.Adam(LEARNING_RATE),
		loss=custom_loss_fn(coeff_scaling_factors),
		metrics=['mae', custom_mae_fn(coeff_scaling_factors)]
	)

	early_stopping_callback = tf.keras.callbacks.EarlyStopping(
		monitor='val_loss',
		patience=5,				# stop if no improvement for 5 epochs
		restore_best_weights=True,
		verbose=1
	)
	save_checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath="_scaled_rel2abs_model_checkpoint.keras",
		save_best_only=True,
		monitor='val_loss',
		verbose=1
	)
	reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
		monitor="val_loss",
		factor=0.5,				# half the LR when plateauing
		patience=2,				# epochs to wait before reducing LR
		min_lr=1e-6,
		verbose=1
	)
	scaled_rel2abs_model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=EPOCHS,
		callbacks=[early_stopping_callback, save_checkpoints_callback, reduce_lr_callback]
	)

	scaled_rel2abs_model.save("_scaled_rel2abs_model.keras")

	# Adds unscaling op to the end to output the final coeffs directly
	rel2abs_model = unscale_rel2abs_model(scaled_rel2abs_model, coeff_scaling_factors)

	export_as_tflite_model(rel2abs_model, "rel2abs_model.tflite")