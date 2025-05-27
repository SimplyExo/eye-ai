#include "ImageUtils.hpp"
#include "Exceptions.hpp"
#include "Log.hpp"
#include "Profiling.hpp"

void check_android_bitmap_result(int result) {
	if (result == ANDROID_BITMAP_RESULT_SUCCESS)
		return;

	switch (result) {
	case ANDROID_BITMAP_RESULT_BAD_PARAMETER:
		LOG_ERROR("Android Bitmap error: Bad Parameter");
		break;
	case ANDROID_BITMAP_RESULT_JNI_EXCEPTION:
		LOG_ERROR("Android Bitmap error: JNI Exception");
		break;
	case ANDROID_BITMAP_RESULT_ALLOCATION_FAILED:
		LOG_ERROR("Android Bitmap error: Allocation failed");
		break;
	default:
		LOG_ERROR("Android Bitmap error: Unknown code: {}", result);
		break;
	}
}

void bitmap_to_rgb_hwc_255_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
) {
	PROFILE_CAMERA_FUNCTION()

	AndroidBitmapInfo info;
	check_android_bitmap_result(AndroidBitmap_getInfo(env, bitmap, &info));

	if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
		throw FormatNotRGBA888Exception(info.format);

	if (out_float_array.size() != (size_t)info.width * (size_t)info.height * 3)
		throw std::invalid_argument("out_float_array");

	void* address_ptr = nullptr;
	check_android_bitmap_result(
		AndroidBitmap_lockPixels(env, bitmap, &address_ptr)
	);
	if (address_ptr == nullptr)
		throw FailedToLockPixelsException();
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

	check_android_bitmap_result(AndroidBitmap_unlockPixels(env, bitmap));
}

void bitmap_to_rgb_chw_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
) {
	PROFILE_CAMERA_FUNCTION()

	AndroidBitmapInfo info;
	check_android_bitmap_result(AndroidBitmap_getInfo(env, bitmap, &info));

	if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
		throw FormatNotRGBA888Exception(info.format);

	if (out_float_array.size() != (size_t)info.width * (size_t)info.height * 3)
		throw std::invalid_argument("out_float_array");

	void* address_ptr = nullptr;
	check_android_bitmap_result(
		AndroidBitmap_lockPixels(env, bitmap, &address_ptr)
	);
	if (address_ptr == nullptr)
		throw FailedToLockPixelsException();
	// RGBA 8888 -> one int for each pixel
	// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
	const auto int_pixels = std::span<int>(
		reinterpret_cast<int*>(address_ptr),
		(size_t)info.width * (size_t)info.height
	);
	// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

	const size_t red_channel_offset = 0;
	const size_t green_channel_offset =
		(size_t)info.width * (size_t)info.height;
	const size_t blue_channel_offset =
		2 * (size_t)info.width * (size_t)info.height;

	for (size_t i = 0; i < (size_t)info.width * (size_t)info.height; i++) {
		const int pixel_color = int_pixels[i];
		out_float_array[red_channel_offset + i] =
			(float)red_channel_from_argb_color(pixel_color) / 255.f;
		out_float_array[green_channel_offset + i] =
			(float)green_channel_from_argb_color(pixel_color) / 255.f;
		out_float_array[blue_channel_offset + i] =
			(float)blue_channel_from_argb_color(pixel_color) / 255.f;
	}

	check_android_bitmap_result(AndroidBitmap_unlockPixels(env, bitmap));
}

void image_bytes_to_argb_int_array(
	const std::span<const jbyte> image_bytes,
	std::span<jint> out_pixels
) {
	PROFILE_CAMERA_FUNCTION()

	if (image_bytes.size_bytes() != out_pixels.size_bytes())
		throw std::invalid_argument("out_pixels");

	size_t i = 0;
	size_t j = 0;
	for (; i < out_pixels.size(); i++) {
		auto r = image_bytes[j++];
		auto g = image_bytes[j++];
		auto b = image_bytes[j++];
		auto a = image_bytes[j++];
		out_pixels[i] = color_argb(a, r, g, b);
	}
}