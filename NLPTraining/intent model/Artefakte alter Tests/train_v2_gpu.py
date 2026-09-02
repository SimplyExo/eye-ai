# train_v2_gpu.py - GPU-optimiertes Training (One-Hot auf GPU)
import tensorflow as tf
import numpy as np

CLASSES = [
    "TEXT_RECOGNITION",
    "OBJECT_DETECTION",
    "CHANGE_SPEECH_SPEED",
    "CHANGE_SPEAKER",
    "REDIRECT_TO_LLM",
    "OPEN_SETTINGS",
    "SET_FREQUENCY",
    "SET_BPS",
    "MEASURE_DISTANCE",
    "ABORT"
]
label_map = {name: i for i, name in enumerate(CLASSES)}

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 100
BATCH_SIZE = 32  # Reduziert für System RAM

print("📁 Lade Dataset...")
def load_dataset(filename: str):
    texts, labels = [], []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            text, label = line.strip().split(";")
            texts.append(text)
            labels.append(label_map[label])
    return texts, np.array(labels, dtype=np.float32)

train_texts, train_labels = load_dataset("DATASET.train")
val_texts, val_labels = load_dataset("DATASET.val")

print(f"📝 Tokenizer ({len(train_texts)} Train, {len(val_texts)} Val)...")
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH
)
vectorize_layer.adapt(train_texts)

vocab = vectorize_layer.get_vocabulary()
with open("vocab.txt", "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(token + "\n")

# Sequenzen einmal berechnen (RAM-freundlich)
print("🔄 Berechne Sequenzen...")
train_sequences = vectorize_layer(np.array(train_texts))
val_sequences = vectorize_layer(np.array(val_texts))

# One-Hot auf GPU (ohne CPU-Kopie)
def preprocess_gpu(sequences, labels):
    # One-Hot direkt auf GPU erstellen
    onehot = tf.one_hot(sequences, VOCAB_SIZE, axis=-1, dtype=tf.float32)
    return onehot, labels

print("🏗️  Baue Dataset (One-Hot auf GPU)...")
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_dataset = train_dataset.map(preprocess_gpu, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE)
# Kein prefetch - verbraucht System RAM

val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_dataset = val_dataset.map(preprocess_gpu, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)

print("🧠 Modell-Architektur:")
inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, VOCAB_SIZE), dtype=tf.float32, name="input_onehot")

x = tf.keras.layers.Conv1D(128, 5, activation='relu', name="conv1")(inputs)
x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)
x = tf.keras.layers.Conv1D(128, 5, activation='relu', name="conv2")(x)
x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)
x = tf.keras.layers.Dense(64, activation='relu', name="dense_1")(x)
x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax', name="output")(x)

model = tf.keras.Model(inputs, outputs, name="nlp_v2_gpu")

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(5e-4),
    metrics=["accuracy"]
)

model.summary()

print(f"\n🚀 Training gestartet (max 20 Epochen)...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)

# Keras Modell speichern (für uint8)
model.save("nlp_v2_final.keras")
print("✅ Keras Modell gespeichert")

# TFLite Konvertierung
def representative_dataset():
    for seq in train_sequences.take(100):
        onehot = tf.one_hot(seq, VOCAB_SIZE, axis=-1).numpy().astype(np.float32)
        yield [onehot.reshape(1, SEQUENCE_LENGTH, VOCAB_SIZE)]

print("\n🔄 TFLite Konvertierung...")

# Dynamic Range
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
tflite_dyn = converter.convert()
with open("nlp_v2_dynamic.tflite", "wb") as f:
    f.write(tflite_dyn)
print("✅ Dynamic Range TFLite gespeichert (nlp_v2_dynamic.tflite)")

# uint8 Quantization
print("🔢 uint8 Quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32  # Softmax braucht float

try:
    tflite_uint8 = converter.convert()
    with open("nlp_v2_uint8.tflite", "wb") as f:
        f.write(tflite_uint8)
    print("✅ uint8 TFLite gespeichert (nlp_v2_uint8.tflite)")
except Exception as e:
    print(f"⚠️  uint8 fehlgeschlagen: {e}")

print("\n✨ Training abgeschlossen!")
print(f"Beste Val Accuracy: {max(history.history['val_accuracy']):.2%}")