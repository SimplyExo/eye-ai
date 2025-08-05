#include "ImageUtils.hpp"
#include "EyeAICore/utils/ImageUtils.hpp"
#include "EyeAICore/utils/Profiling.hpp"

std::optional<BitmapError> check_android_bitmap_result(int result) {
	switch (result) {
	case ANDROID_BITMAP_RESULT_SUCCESS:
		return std::nullopt;
	case ANDROID_BITMAP_RESULT_BAD_PARAMETER:
		return BitmapError("Android Bitmap error: Bad Parameter");
	case ANDROID_BITMAP_RESULT_JNI_EXCEPTION:
		return BitmapError("Android Bitmap error: JNI Exception");
	case ANDROID_BITMAP_RESULT_ALLOCATION_FAILED:
		return BitmapError("Android Bitmap error: Allocation failed");
	default:
		return BitmapError::fmt(
			"Android Bitmap error: Unknown code: {}", result
		);
	}
}

std::optional<BitmapError> bitmap_to_rgb_hwc_255_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array,
	ProfilingFrame& profiling_frame
) {
	PROFILE_FUNCTION(profiling_frame)

	AndroidBitmapInfo info;

	if (const auto error = check_android_bitmap_result(
			AndroidBitmap_getInfo(env, bitmap, &info)
		)) {
		return error;
	}

	if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
		return BitmapError::fmt(
			"bitmap has format {}, but RGBA_8888 was expected", info.format
		);
	}

	if (out_float_array.size() != (size_t)info.width * (size_t)info.height * 3)
		throw std::invalid_argument("out_float_array");

	void* address_ptr = nullptr;
	if (const auto error = check_android_bitmap_result(
			AndroidBitmap_lockPixels(env, bitmap, &address_ptr)
		)) {
		return error;
	}
	if (address_ptr == nullptr) {
		return BitmapError("failed to lock bitmap pixels");
	}
	// RGBA 8888 -> one int for each pixel, lint supression needed because of c
	// api
	// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
	const auto pixel_ptr = std::span<int>(
		reinterpret_cast<int*>(address_ptr),
		(size_t)info.width * (size_t)info.height
	);
	// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

	size_t i = 0;
	size_t j = 0;
	for (; i < (size_t)info.width * (size_t)info.height; i++) {
		const int pixel_color = pixel_ptr[i];
		out_float_array[j++] = (float)red_channel_from_argb_color(pixel_color);
		out_float_array[j++] =
			(float)green_channel_from_argb_color(pixel_color);
		out_float_array[j++] = (float)blue_channel_from_argb_color(pixel_color);
	}

	return check_android_bitmap_result(AndroidBitmap_unlockPixels(env, bitmap));
}