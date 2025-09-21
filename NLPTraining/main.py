# train.py
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

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 250
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
print("✅ Tokenizer-Vokabular gespeichert als vocab.txt")

# -----------------------------
# Texte → Integer-Sequenzen
# -----------------------------
train_sequences = vectorize_layer(np.array(train_texts))
val_sequences = vectorize_layer(np.array(val_texts))

train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# Modell definieren
# -----------------------------
inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.float32)
x = tf.keras.layers.Embedding(VOCAB_SIZE, 64)(inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"]
)

# -----------------------------
# Training
# -----------------------------
model.fit(
    train_dataset,
    epochs=3,
    validation_data=val_dataset
)

# -----------------------------
# Speichern
# -----------------------------
model.save("nlp_model_float32.h5")
print("✅ Keras-Modell gespeichert als nlp_model_float32.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open("nlp_model_int.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite-Modell gespeichert als nlp_model_int.tflite")
