import os
import numpy as np
import sys
from pathlib import Path
from PIL import Image

def get_total_files(dataset_path):
    """Count total number of .png and .npy files in the dataset."""
    count = 0
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.npy')):
                count += 1
    return count

def process_folder(dataset_path, prepared_dataset_path):
    total_files = get_total_files(dataset_path)
    processed_files = 0

    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            output_filepath = os.path.join(prepared_dataset_path, os.path.relpath(filepath, dataset_path))
            os.makedirs(Path(output_filepath).parent.absolute(), exist_ok=True)

            processed_files += 1
            progress = processed_files / total_files * 100
            progress_bar = '[' + '#' * int(progress / 5) + ' ' * (20 - int(progress / 5)) + ']'
            sys.stdout.write(f"\rPreparing: {progress_bar} {progress:.1f}%  {filename[:30]}...{' ' * 10}")
            sys.stdout.flush()

            if filename.lower().endswith('.png'):
                try:
                    with Image.open(filepath) as img:
                        img = img.convert('RGB')  # Ensure 3 channels
                        img = img.resize((256, 256))
                        img_array = np.asarray(img, dtype=np.float32).flatten()
                        bin_path = os.path.splitext(output_filepath)[0] + '_image.bin'
                        img_array.tofile(bin_path)
                except Exception as e:
                    sys.stdout.write(f"\rFailed to process PNG {filename[:30]}...: {e}{' ' * 50}\n")
                    sys.stdout.flush()

            elif filename.lower().endswith('.npy'):
                try:
                    array = np.load(filepath)
                    array.astype(np.float32).flatten().tofile(os.path.splitext(output_filepath)[0] + '.bin')
                except Exception as e:
                    sys.stdout.write(f"\rFailed to process NPY {filename[:30]}...: {e}{' ' * 50}\n")
                    sys.stdout.flush()

    sys.stdout.write(f"\r{' ' * 100}\r")
    sys.stdout.write(f"Preparing {processed_files} files successfully!\n")
    sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_dataset.py <dataset_directory> <prepared_dataset_directory>")
        exit(1)

    folder = sys.argv[1]
    prepared_folder = sys.argv[2]
    process_folder(folder, prepared_folder)