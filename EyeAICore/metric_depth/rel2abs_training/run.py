import sys
import time
import numpy as np
import tensorflow as tf
from dataset import IMG_SIZE

def print_pred_expected_header(index_padding=1):
	print(f"{' ' * index_padding}   Prediction   | Actual       | Diff         | Diff %")

def print_pred_expected(index, predicted, expected):
	diff = predicted - expected
	diff_percentage = abs(diff) / expected
	print(f"[{index}] {predicted:+.5E} | {expected:+.5E} | {diff:+.5E} | {diff_percentage * 100:.2f}%")

def run_rel2abs(rel2abs_model_path, rgbd_images, expected_coeffs):
	"""
	Arguments:
		rel2abs_model_path (str): Path to the TFLite model file.
		rgbd_images (numpy.ndarray): Array of RGB-D images.
		expected_coeffs (numpy.ndarray): Array of expected coefficients.
	"""

	print("Loading model...")
	interpreter = tf.lite.Interpreter(model_path=rel2abs_model_path)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	for i in range(len(rgbd_images)):
		input_rgbd = rgbd_images[i]

		assert input_rgbd.shape == (*IMG_SIZE, 4)

		input = np.expand_dims(input_rgbd, axis=0)

		interpreter.set_tensor(input_details[0]['index'], input)

		start = time.time()
		interpreter.invoke()
		invoke_duration = time.time() - start

		# tensor shape: (1, 5) -> (5)
		pred_coeffs = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

		assert len(pred_coeffs) == len(expected_coeffs[i])

		print(f" === {i} took {invoke_duration:.4f} seconds ===\nCoeffs:")

		print_pred_expected_header()

		for j in range(len(pred_coeffs)):
			print_pred_expected(j, pred_coeffs[j], expected_coeffs[i][j])

		print("\nSamples:")

		pred_func = np.poly1d(pred_coeffs[::-1])
		expected_func = np.poly1d(expected_coeffs[i][::-1])

		print_pred_expected_header(4)

		for j in range(15):
			sample_relative = j * 100 + 100
			pred_abs = pred_func(sample_relative)
			expected_abs = expected_func(sample_relative)
			if expected_abs >= 0:
				print_pred_expected(sample_relative, pred_abs, expected_abs)
			else:
				print(f"[{sample_relative}] ignored due to negative depth, out of realistic range")

		print("")

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python run.py <rgbd_image_paths...>")
		sys.exit(1)

	print("Loading images...")

	input_rgbd_images = []
	expected_coeffs = []
	for i in range(len(sys.argv) - 1):
		input_rgbd_images.append(np.load(sys.argv[i + 1]))
		expected_coeffs.append(np.load(sys.argv[i + 1].replace("rgbd", "coeffs")))

	run_rel2abs("rel2abs_model.tflite", input_rgbd_images, expected_coeffs)
