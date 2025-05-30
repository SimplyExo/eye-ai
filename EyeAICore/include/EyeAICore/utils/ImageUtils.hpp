#pragma once

/// argb 8888 formatted
constexpr int color_argb(int a, int r, int g, int b) {
	return (a << 24) | (r << 16) | (g << 8) | b;
}

/// argb 8888 formatted
constexpr int color_rgb(int r, int g, int b) {
	return color_argb(255, r, g, b);
}

constexpr int red_channel_from_argb_color(int color) {
	return (color >> 16) & 255;
}
constexpr int green_channel_from_argb_color(int color) {
	return (color >> 8) & 255;
}
constexpr int blue_channel_from_argb_color(int color) { return color & 255; }