import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ===== CONFIG =====
IMG_SIZE = (256, 256)	# smaller for speed
N_COEFFS = 5			# 4 degree polynomial
L2_REG = 1e-4

# ===== DUMMY DATASET PLACEHOLDER =====
def load_dataset(batch_size=8):
    rgbd = tf.random.uniform((batch_size, *IMG_SIZE, 4))
    coeffs = tf.random.uniform((batch_size, N_COEFFS), minval=-1, maxval=1)
    ds = tf.data.Dataset.from_tensor_slices((rgbd, coeffs))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = load_dataset()
val_ds = load_dataset()

# ===== MODEL =====
def build_rgbd_regressor(img_size, n_coeffs):
    inputs = layers.Input(shape=(*img_size, 4), name="rgbd_input")

    # Map 4 channels → 3 channels for MobileNet compatibility
    x = layers.Conv2D(3, kernel_size=1, padding="same")(inputs)

    # Use MobileNetV3Small backbone
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(*img_size, 3),
        include_top=False,
        weights=None
    )
    x = base_model(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    outputs = layers.Dense(n_coeffs, activation='linear', name="coeffs_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = build_rgbd_regressor(IMG_SIZE, N_COEFFS)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()

# ===== TRAINING =====
model.fit(train_ds, validation_data=val_ds, epochs=3)

# ===== EXPORT =====
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("rel2abs_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Exported rel2abs_model.tflite")
