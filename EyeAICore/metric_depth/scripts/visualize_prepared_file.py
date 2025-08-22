import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def plot_xy_from_binary(filename):
	data = np.fromfile(filename, dtype=np.float32)
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

	z = np.polyfit(bin_edges, avg_absolute_values, 4)
	p = np.poly1d(z)

	trend_line_function = f"absolute(relative) = {z[0]} * relative⁴ + {z[1]} * relative³ + {z[2]} * relative² + {z[3]} * relative + {z[4]}"
	print(trend_line_function)

	plt.figure(figsize=(8, 5))
	plt.plot(relative_values, absolute_values, alpha=0.25)
	plt.plot(bin_edges, avg_absolute_values, "g-", label="Average")
	plt.plot(relative_values, p(relative_values), "r--", linewidth=5, label=trend_line_function)
	plt.title("relative to absolute Plot")
	plt.xlabel("relative")
	plt.ylabel("absolute")
	plt.grid(True)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path_to_result_file.bin")
        sys.exit(1)

    plot_xy_from_binary(sys.argv[1])
