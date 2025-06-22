import os
import numpy as np
import sys
from pathlib import Path
from PIL import Image

def process_folder(dataset_path, prepared_dataset_path):
	for root, dirs, files in os.walk(dataset_path):
		for filename in files:
			filepath = os.path.join(root, filename)
			output_filepath = os.path.join(prepared_dataset_path, os.path.relpath(filepath, dataset_path))
			os.makedirs(Path(output_filepath).parent.absolute(), exist_ok=True)

			if filename.lower().endswith('.png'):
				try:
					with Image.open(filepath) as img:
						img = img.convert('RGB')  # Ensure 3 channels
						img = img.resize((256, 256))
						img_array = np.asarray(img, dtype=np.float32).flatten()
						bin_path = os.path.splitext(output_filepath)[0] + '_image.bin'
						img_array.tofile(bin_path)
						print(f"Processed PNG: {filename} -> {os.path.basename(bin_path)}")
				except Exception as e:
					print(f"Failed to process PNG {filename}: {e}")

			elif filename.lower().endswith('.npy'):
				try:
					array = np.load(filepath)
					array.astype(np.float32).flatten().tofile(os.path.splitext(output_filepath)[0] + '.bin')
					print(f"Processed NPY: {filename}")
				except Exception as e:
					print(f"Failed to process NPY {filename}: {e}")

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("usage: python prepare_dataset.py <dataset_directory> <prepared_dataset_directory>")
		exit(1)

	folder = sys.argv[1]
	prepared_folder = sys.argv[2]
	process_folder(folder, prepared_folder)