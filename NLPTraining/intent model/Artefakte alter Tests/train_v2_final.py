# train_v2_final.py - Finale Version mit optimiertem Speicher
import tensorflow as tf
import numpy as np
import os

# GPU Speicher begrenzen
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]  # 2GB limit
        )
    except RuntimeError as e:
        print(e)

CLASSES = ["TEXT_RECOGNITION", "OBJECT_DETECTION", "CHANGE_SPEECH_SPEED", "CHANGE_SPEAKER",
           "REDIRECT_TO_LLM", "OPEN_SETTINGS", "SET_FREQUENCY", "SET_BPS", "MEASURE_DISTANCE", "ABORT"]
label_map = {name: i for i, name in enumerate(CLASSES)}

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 100
BATCH_SIZE = 16  # Reduziert für weniger VRAM

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

print(f"📝 Tokenizer...")
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

train_sequences = vectorize_layer(np.array(train_texts))
val_sequences = vectorize_layer(np.array(val_texts))

# Lazy One-Hot (weniger Speicher)
def preprocess_lazy(sequences, labels):
    onehot = tf.one_hot(sequences, VOCAB_SIZE, axis=-1, dtype=tf.float16)  # fp16 statt fp32
    return onehot, labels

# Dataset ohne Prefetch (weniger Buffer)
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_dataset = train_dataset.map(preprocess_lazy, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_dataset = val_dataset.map(preprocess_lazy, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)

print("🏗️  Baue Modell...")
inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, VOCAB_SIZE), dtype=tf.float16, name="input_onehot")

# Einfacheres Modell für weniger VRAM
x = tf.keras.layers.Conv1D(64, 5, activation='relu', name="conv1")(inputs)
x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)
x = tf.keras.layers.Conv1D(64, 3, activation='relu', name="conv2")(x)
x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)
x = tf.keras.layers.Dense(32, activation='relu', name="dense_1")(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax', name="output")(x)

model = tf.keras.Model(inputs, outputs, name="nlp_v2_final")
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(5e-4),
    metrics=["accuracy"]
)

model.summary()

print(f"\n🚀 Training ({len(train_texts)} Samples)...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
]

model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)

# Keras speichern
model.save("nlp_v2_final.keras")
print("✅ Keras Modell gespeichert")

# TFLite
def representative_dataset():
    for seq in train_sequences.take(50):
        onehot = tf.one_hot(seq, VOCAB_SIZE, axis=-1).numpy().astype(np.float32)
        yield [onehot.reshape(1, SEQUENCE_LENGTH, VOCAB_SIZE)]

print("\n🔄 TFLite Konvertierung...")

# Dynamic
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_dyn = converter.convert()
with open("nlp_v2_dynamic.tflite", "wb") as f:
    f.write(tflite_dyn)
print("✅ Dynamic TFLite gespeichert")

# uint8
print("🔢 uint8 Quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32

try:
    tflite_uint8 = converter.convert()
    with open("nlp_v2_uint8.tflite", "wb") as f:
        f.write(tflite_uint8)
    print("✅ uint8 TFLite gespeichert")
except Exception as e:
    print(f"⚠️  uint8 fehlgeschlagen: {e}")

print("\n✨ Fertig!")
