import numpy as np
import tensorflow as tf
from ultralytics import YOLO

MODEL_NAME = "yolov8n"

print(f"🚀 1. Lade YOLOv8 Modell ({MODEL_NAME}.pt)...")
model = YOLO(f"{MODEL_NAME}.pt")

print("\n🔄 2. Exportiere SavedModel...")
# Standard-Export ohne interne Kalibrierung
saved_model_path = model.export(format="saved_model")

print("\n⚡ 3. Konvertiere für QNN Delegate (HTP)...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter._experimental_disable_per_channel_quantization_for_dense_layers = True

def qnn_representative_dataset():
    for _ in range(100):
        dummy_img = np.random.randint(0, 256, size=(1, 640, 640, 3), dtype=np.uint8)
        yield [dummy_img.astype(np.float32)]

converter.representative_dataset = qnn_representative_dataset

print("\n⚙️ Quantisiere TFLite-Graph...")
tflite_model = converter.convert()

output_filename = f"{MODEL_NAME}_qnn_htp.tflite"
with open(output_filename, "wb") as f:
    f.write(tflite_model)

print(f"\n✅ FERTIG! Fertiges Modell: {output_filename}")
