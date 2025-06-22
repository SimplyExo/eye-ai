import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def plot_xy_from_binary(filename):
	data = pd.read_csv(filename, header=None)
	data.columns = ['relative', 'absolute']
	data = data.sort_values(by='relative')

	z = np.polyfit(data['relative'], data['absolute'], 4)
	p = np.poly1d(z)

	trend_line_function = f"absolute(relative) = {z[0]} * relative⁴ + {z[1]} * relative³ + {z[2]} * relative² + {z[3]} * relative + {z[4]}"
	print(trend_line_function)

	bin_width = 10
	bins = np.arange(data['relative'].min(), data['relative'].max() + bin_width, bin_width)
	data['relative_bin'] = pd.cut(data['relative'], bins)

	avg_data = data.groupby('relative_bin').agg({'relative': 'mean', 'absolute': 'mean'}).reset_index(drop=True)

	plt.figure(figsize=(8, 5))
	plt.plot(data['relative'], data['absolute'], alpha=0.25)
	plt.plot(avg_data['relative'], avg_data['absolute'], "g-", label="Average")
	plt.plot(data['relative'], p(data['relative']), "r--", linewidth=5, label=trend_line_function)
	plt.title("relative to absolute Plot")
	plt.xlabel("relative")
	plt.ylabel("absolute")
	plt.grid(True)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path_to_result_file.csv")
        sys.exit(1)

    plot_xy_from_binary(sys.argv[1])
