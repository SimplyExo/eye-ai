import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataset import IMG_SIZE, TRAIN_VAL_RATIO

def eval_rel2abs(rel2abs_coeffs, rgbd_input, rel_abs_pairs):
	assert rgbd_input.shape == (*IMG_SIZE, 4)

	raw_relative_depths = rel_abs_pairs[0::2]
	expected_absolute_depths = rel_abs_pairs[1::2]

	predicted_absolute_depths = np.polyval(rel2abs_coeffs[::-1], raw_relative_depths)

	return expected_absolute_depths, predicted_absolute_depths

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python run.py <dataset_path>")
		sys.exit(1)

	rel2abs_coeffs = np.load("rel2abs_coeffs.npy")

	dataset_path = sys.argv[1]

	ds_rgbd_image_count = len([
		file for file in os.listdir(dataset_path)
		if file.endswith("_rgbd.npy")
	])

	train_count = int(ds_rgbd_image_count * TRAIN_VAL_RATIO)

	val_indices = np.arange(train_count, ds_rgbd_image_count)

	avg_error_list = []

	for i in val_indices:
		input_rgbd_image = np.load(os.path.join(dataset_path, f"{i}_rgbd.npy"))
		rel_abs_pairs = np.load(os.path.join(dataset_path, f"{i}_rel_abs_pairs.npy"))

		true, pred = eval_rel2abs(rel2abs_coeffs, input_rgbd_image, rel_abs_pairs)
		avg_error = np.mean(np.abs(true - pred))

		median_error = np.median(np.abs(true - pred))

		avg_error_list.append(avg_error)

		print(f"image {i}: avg_error: {avg_error:.4f} meters, median_error: {median_error:.4f} meters")

	avg_errors = np.array(avg_error_list)

	np.save('avg_errors.npy', avg_errors)

	avg_across_ds = np.mean(avg_errors)
	avg_errors_bins = np.arange(avg_errors.min(), avg_errors.max() + 0.05, 0.05)
	avg_errors_bin_counts, avg_errors_bin_edges = np.histogram(avg_errors, bins=avg_errors_bins)
	median_avg_error_ds = avg_errors_bin_edges[np.argmax(avg_errors_bin_counts)]
	print(f"Average error across dataset: {avg_across_ds:.4f} meters")
	print(f"Median avg error across dataset: {median_avg_error_ds:.4f} meters")

	plt.hist(avg_errors, bins=avg_errors_bins, edgecolor="black", alpha=0.7)

	plt.xlabel("Avg error per image (meters)")
	plt.ylabel("Count")
	plt.title("Per image avg error distribution in dataset")
	plt.grid(True, linestyle="--", alpha=0.5)

	plt.savefig('avg_error_distribution.png')