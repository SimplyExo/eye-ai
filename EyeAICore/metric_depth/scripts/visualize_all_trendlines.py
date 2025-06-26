import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_result_file_and_find_coeffs(filepath):
    try:
        data = np.fromfile(filepath, dtype=np.float32)
        unsorted_relative_values = data[::2]
        unsorted_absolute_values = data[1::2]
        sort_indices = np.argsort(unsorted_relative_values)
        relative_values = unsorted_relative_values[sort_indices]
        absolute_values = unsorted_absolute_values[sort_indices]

        num_bins = 20
        bin_edges = np.linspace(relative_values.min(), relative_values.max(), num_bins)
        bin_indices = np.digitize(relative_values, bin_edges) - 1
        bin_indices[bin_indices == num_bins] = num_bins - 1
        avg_absolute_values = np.array([
            absolute_values[bin_indices == i].mean()
            for i in range(num_bins)
        ])
        coeffs = np.polyfit(bin_edges, avg_absolute_values, 4)
        return coeffs
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return None

def write_coeffs(directory):
    with open(directory + "/coeffs.csv", "w") as f:
        for filename in os.listdir(directory):
            if not filename.lower().endswith(".bin"):
                continue
            filepath = os.path.join(directory, filename)
            coeffs = load_result_file_and_find_coeffs(filepath)
            if coeffs is None:
                continue
            f.write(",".join(str(coeff) for coeff in coeffs) + "\n")
        f.close()

def plot_coeffs(directory):
    coeffs_df = pd.read_csv(directory + '/coeffs.csv', header=None, names=['a', 'b', 'c', 'd', 'e'])

    x = np.linspace(0, 1200, 1200)

    plt.figure(figsize=(10, 6))

    avg_y = np.zeros(dtype=float, shape=(len(x),))
    avg_y_count = 0

    for idx, row in coeffs_df.iterrows():
        a, b, c, d, e = row
        f = np.poly1d(np.array([a, b, c, d, e]))
        y = f(x)

        plt.plot(x, y, alpha=0.15)
        avg_y += y
        avg_y_count += 1

    avg_y /= avg_y_count

    plt.plot(x, avg_y, label='Average', color='red', linestyle='--', linewidth=2)

    best_fitting_coeffs = np.polyfit(x, avg_y, 4)
    print(f"Best fitting coefficients: {best_fitting_coeffs}")

    plt.title("Cubic Polynomials from CSV Coefficients")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot trend lines from CSV files in a directory")
    parser.add_argument("directory", help="Path to directory containing CSV files")
    args = parser.parse_args()

    write_coeffs(args.directory)
    plot_coeffs(args.directory)