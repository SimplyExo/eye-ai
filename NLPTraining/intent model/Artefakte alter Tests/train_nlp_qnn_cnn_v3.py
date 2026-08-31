# train_nlp_qnn_cnn_v3.py - Verbessertes NPU-Training
import tensorflow as tf
import numpy as np
import subprocess

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

VOCAB_SIZE = 10000  # Voller Vokabular
SEQUENCE_LENGTH = 250  # Vollständige Sequenzlänge
BATCH_SIZE = 64

# -----------------------------
# Dataset laden
# -----------------------------
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

# -----------------------------
# Tokenizer vorbereiten
# -----------------------------
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH
)
vectorize_layer.adapt(train_texts)

# Tokenizer-Vokabular speichern
vocab = vectorize_layer.get_vocabulary()
with open("vocab.txt", "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(token + "\n")
print(f"✅ Tokenizer-Vokabular gespeichert ({len(vocab)} Tokens)")

# -----------------------------
# Texte → Integer-Sequenzen
# -----------------------------
train_sequences = vectorize_layer(np.array(train_texts))
val_sequences = vectorize_layer(np.array(val_texts))

# -----------------------------
# One-Hot Encoding (NPU-kompatibel)
# -----------------------------
def preprocess_for_model(sequences):
    return tf.one_hot(sequences, VOCAB_SIZE, axis=-1)

# Trainingsdaten mit One-Hot
train_onehot = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_onehot = train_onehot.map(lambda x, y: (preprocess_for_model(x), y))
train_onehot = train_onehot.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_onehot = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_onehot = val_onehot.map(lambda x, y: (preprocess_for_model(x), y))
val_onehot = val_onehot.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# Verbessertes CNN-Modell
# -----------------------------
inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, VOCAB_SIZE), dtype=tf.float32, name="input_onehot")

# Mehr Filter und Layer
x = tf.keras.layers.Conv1D(256, 7, activation='relu', padding='same', name="conv1")(inputs)
x = tf.keras.layers.BatchNormalization(name="bn1")(x)
x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)

x = tf.keras.layers.Conv1D(256, 5, activation='relu', padding='same', name="conv2")(x)
x = tf.keras.layers.BatchNormalization(name="bn2")(x)
x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)

x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same', name="conv3")(x)
x = tf.keras.layers.BatchNormalization(name="bn3")(x)

x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)

x = tf.keras.layers.Dense(128, activation='relu', name="dense_1")(x)
x = tf.keras.layers.Dropout(0.5, name="dropout_1")(x)
x = tf.keras.layers.Dense(64, activation='relu', name="dense_2")(x)
x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax', name="output")(x)

model = tf.keras.Model(inputs, outputs, name="nlp_classifier_cnn_v3")

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Training mit Callbacks
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
]

print("\n=== Training gestartet ===")
model.fit(
    train_onehot,
    epochs=50,
    validation_data=val_onehot,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# TFLite-Konvertierung
# -----------------------------
def representative_dataset():
    for seq in train_sequences.take(200):
        onehot = tf.one_hot(seq, VOCAB_SIZE, axis=-1).numpy().astype(np.float32)
        yield [onehot.reshape(1, SEQUENCE_LENGTH, VOCAB_SIZE)]

print("\n=== TFLite Konvertierung ===")

# Dynamic Range Quantization (empfohlen für QNN NPU)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
tflite_dyn_model = converter.convert()
with open("nlp_model_v3_dynamic.tflite", "wb") as f:
    f.write(tflite_dyn_model)
print("✅ Dynamic Range TFLite gespeichert (nlp_model_v3_dynamic.tflite)")

# Float16
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_fp16_model = converter.convert()
with open("nlp_model_v3_float16.tflite", "wb") as f:
    f.write(tflite_fp16_model)
print("✅ Float16 TFLite gespeichert (nlp_model_v3_float16.tflite)")

# -----------------------------
# Verifizierung
# -----------------------------
print("\n=== NPU-Kompatibilität prüfen ===")
for model_name in ['nlp_model_v3_dynamic.tflite', 'nlp_model_v3_float16.tflite']:
    print(f'\n🔍 {model_name}:')
    try:
        interpreter = tf.lite.Interpreter(model_path=model_name)
        interpreter.allocate_tensors()
        with open(model_name, 'rb') as f:
            content = f.read()
            has_select = b'SelectV2' in content
            has_flex = b'Flex' in content
            print(f"  {'⚠️' if has_select else '✅'} SelectV2: {'Ja' if has_select else 'Nein'}")
            print(f"  {'⚠️' if has_flex else '✅'} Flex Ops: {'Ja' if has_flex else 'Nein'}")
    except Exception as e:
        print(f'  ❌ Fehler: {e}')

print("\n✨ Training abgeschlossen!")
print("   Verwende nlp_model_v3_dynamic.tflite für QNN NPU.")