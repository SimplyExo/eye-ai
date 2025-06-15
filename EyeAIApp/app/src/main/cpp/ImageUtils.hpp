#pragma once

#include <tl/expected.hpp>
#include <android/bitmap.h>
#include <span>

[[nodiscard]] tl::expected<void, std::string> check_android_bitmap_result(int result);

/// converts pixel from bitmap into float array with (height, width, channel)
/// shape and 3 rgb-channels each in the range of 0.0f to 255.0f
/// often the right format for use with tflite models
[[nodiscard]] tl::expected<void, std::string> bitmap_to_rgb_hwc_255_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
);

/// converts pixel from bitmap into float array with (channel, height, width)
/// shape and 3 rgb-channels each in the range of 0.0f to 1.0f
/// often the right format for use with onnx models
[[nodiscard]] tl::expected<void, std::string> bitmap_to_rgb_chw_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
);

/// image_bytes should have 4 bytes (4 argb channels) for each pixel
[[nodiscard]] tl::expected<void, std::string> image_bytes_to_argb_int_array(
	std::span<const jbyte> image_bytes,
	std::span<jint> out_pixels
);