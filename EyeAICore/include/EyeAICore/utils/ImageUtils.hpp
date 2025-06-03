#pragma once

#include <cctype>

/// argb 8888 formatted
constexpr int color_argb(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
	return (a << 24) | (r << 16) | (g << 8) | b;
}

/// argb 8888 formatted
constexpr int color_rgb(uint8_t r, uint8_t g, uint8_t b) {
	return color_argb(255, r, g, b);
}

constexpr uint8_t red_channel_from_argb_color(int color) {
	return (color >> 16) & 255;
}
constexpr uint8_t green_channel_from_argb_color(int color) {
	return (color >> 8) & 255;
}
constexpr uint8_t blue_channel_from_argb_color(int color) {
	return color & 255;
}