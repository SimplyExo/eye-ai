import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, optimizers, saving, losses
from dataset import load_dataset, IMG_SIZE, N_COEFFS
from model import ScaledRel2AbsModel

FROZEN_LEARNING_RATE = 1e-3
FROZEN_EPOCHS = 3
UNFROZEN_LEARNING_RATE = 3e-5
UNFROZEN_EPOCHS = 10

def train_scaled_rel2abs_model(scaled_rel2abs_model, checkpoint_filepath, epochs, lr):
	scaled_rel2abs_model.compile(
		optimizer=optimizers.Adam(lr, clipnorm=1.0),
		loss=None,
		metrics=[]
	)
	early_stopping_callback = tf.keras.callbacks.EarlyStopping(
		monitor='val_loss',
		patience=5,				# stop if no improvement for 5 epochs
		restore_best_weights=True,
		verbose=1
	)
	save_checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
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
		epochs=epochs,
		callbacks=[early_stopping_callback, save_checkpoints_callback, reduce_lr_callback]
	)

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
	# converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()

	with open(tflite_filepath, "wb") as f:
		f.write(tflite_model)

	print(f"Exported {tflite_filepath}")

if __name__ == "__main__":
	if len(sys.argv) != 2 and len(sys.argv) != 4:
		print("Usage: python train.py <path_to_prepared_dataset> <optional: --export-checkpoint checkpoint_filepath>")
		sys.exit(1)

	dataset_root_path = sys.argv[1]

	train_ds, val_ds, coeff_scaling_factors = load_dataset(dataset_root_path)

	scaled_rel2abs_model = ScaledRel2AbsModel(coeff_scaling_factors)

	export = len(sys.argv) == 4 and sys.argv[2] == "--export-checkpoint"

	if export:
		checkpoint_filepath = sys.argv[3]
		scaled_rel2abs_model.load_weights(checkpoint_filepath)
		scaled_rel2abs_model.trainable = False
		dummy_input = tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 4))
		_ = scaled_rel2abs_model(dummy_input, training=False)
	else:
		train_scaled_rel2abs_model(scaled_rel2abs_model, "_scaled_rel2abs_model_checkpoint.h5", FROZEN_EPOCHS, FROZEN_LEARNING_RATE)
		scaled_rel2abs_model.base_model.trainable = True
		train_scaled_rel2abs_model(scaled_rel2abs_model, "_scaled_rel2abs_model_checkpoint.h5", UNFROZEN_EPOCHS, UNFROZEN_LEARNING_RATE)

	# Adds unscaling op to the end to output the final coeffs directly
	rel2abs_model = unscale_rel2abs_model(scaled_rel2abs_model, coeff_scaling_factors)

	# Make batch_size fixed to 1 for inference
	fixed_rgbd_input = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 4), batch_size=1, name="rgbd_input")
	fixed_rel2abs_model = tf.keras.Model(inputs=fixed_rgbd_input, outputs=rel2abs_model(fixed_rgbd_input))

	try:
		export_as_tflite_model(fixed_rel2abs_model, "rel2abs_model.tflite")
	except Exception as e:
		print(f"Error exporting model to TFLite: {e}")
