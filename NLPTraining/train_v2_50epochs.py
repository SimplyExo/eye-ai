import tensorflow as tf
import numpy as np

CLASSES = ["TEXT_RECOGNITION", "OBJECT_DETECTION", "CHANGE_SPEECH_SPEED", "CHANGE_SPEAKER",
           "REDIRECT_TO_LLM", "OPEN_SETTINGS", "SET_FREQUENCY", "SET_BPS", "MEASURE_DISTANCE", "ABORT"]
label_map = {name: i for i, name in enumerate(CLASSES)}

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 100
BATCH_SIZE = 16

print("📁 Dataset...")
def load_dataset(filename):
    texts, labels = [], []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            text, label = line.strip().split(";")
            texts.append(text)
            labels.append(label_map[label])
    return texts, np.array(labels, dtype=np.float32)

train_texts, train_labels = load_dataset("DATASET.train")
val_texts, val_labels = load_dataset("DATASET.val")

print("📝 Tokenizer...")
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

print("🔄 Sequenzen...")
train_sequences = vectorize_layer(np.array(train_texts))
val_sequences = vectorize_layer(np.array(val_texts))

def preprocess_gpu(sequences, labels):
    onehot = tf.one_hot(sequences, VOCAB_SIZE, axis=-1)
    return onehot, labels

train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_dataset = train_dataset.map(preprocess_gpu)
train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_dataset = val_dataset.map(preprocess_gpu)
val_dataset = val_dataset.batch(BATCH_SIZE)

print("🏗️ Modell...")
inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, VOCAB_SIZE), dtype=tf.float32)
x = tf.keras.layers.Conv1D(128, 5, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Conv1D(128, 5, activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-3), metrics=["accuracy"])

print("\n🚀 Training 50 Epochen (KEIN Early Stopping)...")
model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    verbose=1
)

model.save("nlp_v2_50epochs.keras")
print("\n✅ Keras Modell gespeichert")

def representative_dataset():
    for i in range(50):
        seq = train_sequences[i]
        onehot = tf.one_hot(seq, VOCAB_SIZE, axis=-1).numpy().astype(np.float32)
        yield [onehot.reshape(1, SEQUENCE_LENGTH, VOCAB_SIZE)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_dyn = converter.convert()
with open("nlp_v2_50epochs_dynamic.tflite", "wb") as f:
    f.write(tflite_dyn)

max_val_acc = max(model.history.history['val_accuracy'])
print(f"📊 Final Val Accuracy: {max_val_acc*100:.1f}%")
print("✨ Fertig!")