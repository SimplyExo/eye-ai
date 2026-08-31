# train_nlp_qnn_cnn_v2_extended.py - v2 mit mehr Epochen
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
SEQUENCE_LENGTH = 250  # Erhöht auf 250
BATCH_SIZE = 64

def load_dataset(filename: str):
    texts, labels = [], []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            text, label = line.strip().split(";")
            texts.append(text)
            labels.append(label_map[label])
    return texts, np.array(labels, dtype=np.float32)

print("📁 Lade Dataset...")
train_texts, train_labels = load_dataset("DATASET.train")
val_texts, val_labels = load_dataset("DATASET.val")

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
print(f"✅ Vokabular ({len(vocab)} Tokens)")

train_sequences = vectorize_layer(np.array(train_texts))
val_sequences = vectorize_layer(np.array(val_texts))

def preprocess_for_model(sequences):
    return tf.one_hot(sequences, VOCAB_SIZE, axis=-1)

train_onehot = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_onehot = train_onehot.map(lambda x, y: (preprocess_for_model(x), y))
train_onehot = train_onehot.shuffle(5000).batch(BATCH_SIZE)

val_onehot = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_onehot = val_onehot.map(lambda x, y: (preprocess_for_model(x), y))
val_onehot = val_onehot.batch(BATCH_SIZE)

inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, VOCAB_SIZE), dtype=tf.float32, name="input_onehot")

x = tf.keras.layers.Conv1D(128, 5, activation='relu', name="conv1")(inputs)
x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)
x = tf.keras.layers.Conv1D(128, 5, activation='relu', name="conv2")(x)
x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)
x = tf.keras.layers.Dense(64, activation='relu', name="dense_1")(x)
x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax', name="output")(x)

model = tf.keras.Model(inputs, outputs, name="nlp_classifier_cnn_v2_extended")

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
]

print("\n=== Training gestartet (max 50 Epochen) ===")
model.fit(
    train_onehot,
    epochs=50,
    validation_data=val_onehot,
    callbacks=callbacks,
    verbose=1
)

model.save("nlp_model_v2_extended.keras")
print("✅ Keras Modell gespeichert")

def representative_dataset():
    for seq in train_sequences.take(100):
        onehot = tf.one_hot(seq, VOCAB_SIZE, axis=-1).numpy().astype(np.float32)
        yield [onehot.reshape(1, SEQUENCE_LENGTH, VOCAB_SIZE)]

print("\n=== TFLite Konvertierung ===")

# Dynamic Range
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
tflite_dyn_model = converter.convert()
with open("nlp_model_v2_dynamic.tflite", "wb") as f:
    f.write(tflite_dyn_model)
print("✅ Dynamic Range TFLite gespeichert")

# uint8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32

try:
    tflite_uint8_model = converter.convert()
    with open("nlp_model_v2_uint8.tflite", "wb") as f:
        f.write(tflite_uint8_model)
    print("✅ Full Integer TFLite gespeichert")
except Exception as e:
    print(f"⚠️  Full Integer fehlgeschlagen: {e}")

print("\n✨ Training abgeschlossen!")
