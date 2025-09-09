import sys
import numpy as np
from PIL import Image

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python split_rgbd.py path/to/rgbd.npy")
		sys.exit(1)

	filepath = sys.argv[1]
	rgbd = np.load(filepath)

	rgb = rgbd[:, :, :3]
	depth = rgbd[:, :, 3:]

	# Normalize RGB to [0,255]
	rgb_8bit = (((rgb + 1.0) / 2.0) * 255.0).clip(0, 255).astype(np.uint8)

	# Normalize Depth to [0,65535]
	depth_16bit = ((depth + 1.0) / 2.0 * 65535.0).clip(0, 65535).astype(np.uint16)

	Image.fromarray(rgb_8bit, mode="RGB").save("rgb.png")

	Image.fromarray(depth_16bit, mode="I;16").save("depth.png")