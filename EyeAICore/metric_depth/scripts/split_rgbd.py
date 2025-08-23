import sys
import numpy as np
from PIL import Image

filepath = sys.argv[1]
rgbd = np.load(filepath)

rgb = rgbd[:, :, :3]
depth = rgbd[:, :, 3:]

# Normalize RGB to [0,255]
rgb_8bit = (((rgb + 1.0) / 2.0) * 255.0).clip(0, 255).astype(np.uint8)

# Normalize Depth to [0,65535]
depth_16bit = ((depth + 1.0) / 2.0 * 65535.0).clip(0, 65535).astype(np.uint16)

# Save RGB PNG
Image.fromarray(rgb_8bit, mode="RGB").save("rgb.png")

# Save Depth PNG (16-bit grayscale, mode "I;16")
Image.fromarray(depth_16bit, mode="I;16").save("depth.png")