import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def load_csv_and_find_coeff(filepath):
    try:
        data = pd.read_csv(filepath, header=None)
        data.columns = ['relative', 'absolute']
        data = data.sort_values(by='relative')
        coeff = np.polyfit(data['relative'], data['absolute'], 4)
        p = np.poly1d(coeff)
        return coeff
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return None

def write_coeffs(directory):
    with open(directory + "/coeffs.csv", "w") as f:
        for filename in os.listdir(directory):
            if not filename.lower().endswith(".csv"):
                continue
            if filename.startswith("coeffs.csv"):
                continue
            if "outdoor" in filename:
                continue
            filepath = os.path.join(directory, filename)
            coeffs = load_csv_and_find_coeff(filepath)
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
        y = f(x)# a * x**4 + b * x**3 + c * x**2 + d * x + e
        if np.any(y > 7.5) or np.any(y < -2.5):
            continue

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