import tensorflow as tf
from tensorflow.keras import layers, optimizers, saving, losses
from dataset import load_dataset, IMG_SIZE, N_COEFFS


def custom_loss(scaled_pred_coeffs, coeff_scaling_factors, rel_abs_pairs):
	a = coeff_scaling_factors[:, 0]
	b = coeff_scaling_factors[:, 1]

	unscaled_coeffs = (scaled_pred_coeffs - b) / a

	powers = tf.range(N_COEFFS, dtype=tf.float32)

	rel = rel_abs_pairs[:, :, 0:1]
	abs_true = rel_abs_pairs[:, :, 1:2]

	rel_powers = tf.pow(rel, powers)

	unscaled_coeffs_exp = tf.expand_dims(unscaled_coeffs, axis=1)

	abs_pred = tf.reduce_sum(unscaled_coeffs_exp * rel_powers, axis=-1, keepdims=True)

	return tf.reduce_mean(tf.square(abs_true - abs_pred))

class ScaledRel2AbsModel(tf.keras.Model):
	def __init__(self, coeff_scaling_factors):
		super().__init__()

		self.coeff_scaling_factors = coeff_scaling_factors

		self.loss_tracker = tf.keras.metrics.Mean(name="loss")
		self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

		# Inputs
		inputs = layers.Input(shape=(*IMG_SIZE, 4), name="rgbd_input")
		rgb = layers.Lambda(lambda x: x[..., :3])(inputs)   # first 3 channels
		depth = layers.Lambda(lambda x: x[..., 3:4])(inputs) # last channel

		# --- RGB branch (pretrained MobileNetV3Large) ---
		self.base_model = tf.keras.applications.MobileNetV3Large(
			input_shape=(*IMG_SIZE, 3),
			include_top=False,
			weights="imagenet"
		)
		self.base_model.trainable = False
		rgb_features = self.base_model(rgb, training=False)
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

		self.model = tf.keras.models.Model(inputs=inputs, outputs=scaled_coeffs_output)


	@property
	def metrics(self):
		# Keras will reset these each epoch
		return [self.loss_tracker, self.val_loss_tracker]

	def train_step(self, data):
		rgbd_images, rel_abs_pairs_batch = data
		with tf.GradientTape() as tape:
			coeffs = self(rgbd_images, training=True)
			loss = custom_loss(coeffs, self.coeff_scaling_factors, rel_abs_pairs_batch)
		grads = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		self.loss_tracker.update_state(loss)
		return {"loss": self.loss_tracker.result()}

	def test_step(self, data):
		rgbd_images, rel_abs_pairs_batch = data
		coeffs = self(rgbd_images, training=False)
		loss = custom_loss(coeffs, self.coeff_scaling_factors, rel_abs_pairs_batch)

		# Update tracker
		self.val_loss_tracker.update_state(loss)
		return {"loss": self.val_loss_tracker.result()}


	def call(self, inputs):
		return self.model(inputs)
