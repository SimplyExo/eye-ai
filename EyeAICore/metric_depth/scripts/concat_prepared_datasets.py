import os
import sys
import re
import shutil
import numpy as np

def copy_dataset(dataset_dir, id_offset, output_prepared_dataset):
    """
    Returns:
        highest_id: int
    """

    # regex to match filenames like "123_coeffs.npy"
    coeffs_pattern = re.compile(r"(\d+)\_coeffs\.npy$")

    # regex to match filenames like "123_rel_abs_pairs.npy"
    rel_abs_pairs_pattern = re.compile(r"(\d+)\_rel_abs_pairs\.npy$")

    # regex to match filenames like "123_rgbd.npy"
    rgbd_pattern = re.compile(r"(\d+)\_rgbd\.npy$")

    highest_id = -1

    for filename in os.listdir(dataset_dir):
        coeffs_match = coeffs_pattern.match(filename)
        rel_abs_pairs_match = rel_abs_pairs_pattern.match(filename)
        rgbd_match = rgbd_pattern.match(filename)

        if coeffs_match:
            id = int(coeffs_match.group(1))
            new_id = id + id_offset
            new_filename = f"{new_id}_coeffs.npy"
            src_path = os.path.join(dataset_dir, filename)
            dst_path = os.path.join(output_prepared_dataset, new_filename)
            print(f"Copying {filename}       -> {new_filename}")
            shutil.copyfile(src_path, dst_path)

        if rel_abs_pairs_match:
            id = int(rel_abs_pairs_match.group(1))
            new_id = id + id_offset
            new_filename = f"{new_id}_rel_abs_pairs.npy"
            src_path = os.path.join(dataset_dir, filename)
            dst_path = os.path.join(output_prepared_dataset, new_filename)
            print(f"Copying {filename}               -> {new_filename}")
            shutil.copyfile(src_path, dst_path)

        if rgbd_match:
            id = int(rgbd_match.group(1))
            new_id = id + id_offset
            if new_id > highest_id:
                highest_id = new_id
            new_filename = f"{new_id}_rgbd.npy"
            src_path = os.path.join(dataset_dir, filename)
            dst_path = os.path.join(output_prepared_dataset, new_filename)
            print(f"Copying {filename}     -> {new_filename}")
            shutil.copyfile(src_path, dst_path)

    return highest_id

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python concat_prepared_datasets.py <dataset_a> <dataset_b> <output_prepared_dataset>")
        sys.exit(1)

    dataset_a_root = sys.argv[1]
    dataset_b_root = sys.argv[2]
    output_prepared_dataset = sys.argv[3]

    os.makedirs(output_prepared_dataset, exist_ok=True)

    highest_id_a = copy_dataset(dataset_a_root, 0, output_prepared_dataset)
    copy_dataset(dataset_b_root, highest_id_a + 1, output_prepared_dataset)

    # Combine raw relative depth sample from both datasets
    raw_relative_depth_samples_a = np.load(f"{dataset_a_root}/raw_relative_depth_samples.npy")
    raw_relative_depth_samples_b = np.load(f"{dataset_b_root}/raw_relative_depth_samples.npy")

    combined_raw_relative_depth_samples = raw_relative_depth_samples_a
    for i in range(int(len(raw_relative_depth_samples_b) / 2)):
        combined_raw_relative_depth_samples[i * 2] = raw_relative_depth_samples_b[i * 2]
    np.save(f"{output_prepared_dataset}/raw_relative_depth_samples.npy", combined_raw_relative_depth_samples)
    print(f"Combined raw relative depth samples saved to {output_prepared_dataset}/raw_relative_depth_samples.npy")