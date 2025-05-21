#pragma once

#include <android/bitmap.h>
#include <span>

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

void check_android_bitmap_result(int result);

/// converts pixel from bitmap into float array with (height, width, channel)
/// shape and 3 rgb-channels each in the range of 0.0f to 255.0f
/// often the right format for use with tflite models
void bitmap_to_rgb_hwc_255_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
);

/// converts pixel from bitmap into float array with (channel, height, width)
/// shape and 3 rgb-channels each in the range of 0.0f to 1.0f
/// often the right format for use with onnx models
void bitmap_to_rgb_chw_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
);

/// image_bytes should have 4 bytes (4 argb channels) for each pixel
void image_bytes_to_argb_int_array(
	std::span<const jbyte> image_bytes,
	std::span<jint> out_pixels
);