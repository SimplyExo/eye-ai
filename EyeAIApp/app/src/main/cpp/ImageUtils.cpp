#include "ImageUtils.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/ImageUtils.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "Log.hpp"

tl::expected<void, std::string> check_android_bitmap_result(int result) {
	switch (result) {
	case ANDROID_BITMAP_RESULT_SUCCESS:
		return {};
	case ANDROID_BITMAP_RESULT_BAD_PARAMETER:
		return tl::unexpected("Android Bitmap error: Bad Parameter");
	case ANDROID_BITMAP_RESULT_JNI_EXCEPTION:
		return tl::unexpected("Android Bitmap error: JNI Exception");
	case ANDROID_BITMAP_RESULT_ALLOCATION_FAILED:
		return tl::unexpected("Android Bitmap error: Allocation failed");
	default:
		return tl::unexpected_fmt(
			"Android Bitmap error: Unknown code: {}", result
		);
	}
}

tl::expected<void, std::string> bitmap_to_rgb_hwc_255_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
) {
	PROFILE_CAMERA_FUNCTION()

	AndroidBitmapInfo info;
	auto result =
		check_android_bitmap_result(AndroidBitmap_getInfo(env, bitmap, &info));
	if (!result.has_value())
		return tl::unexpected(result.error());

	if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
		return tl::unexpected_fmt(
			"bitmap has format {}, but RGBA_8888 was expected", info.format
		);
	}

	if (out_float_array.size() != (size_t)info.width * (size_t)info.height * 3)
		throw std::invalid_argument("out_float_array");

	void* address_ptr = nullptr;
	result = check_android_bitmap_result(
		AndroidBitmap_lockPixels(env, bitmap, &address_ptr)
	);
	if (!result.has_value())
		return tl::unexpected(result.error());
	if (address_ptr == nullptr) {
		return tl::unexpected("failed to lock bitmap pixels");
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

tl::expected<void, std::string> bitmap_to_rgb_chw_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
) {
	PROFILE_CAMERA_FUNCTION()

	AndroidBitmapInfo info;
	auto result =
		check_android_bitmap_result(AndroidBitmap_getInfo(env, bitmap, &info));
	if (!result.has_value())
		return tl::unexpected(result.error());

	if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
		return tl::unexpected_fmt(
			"bitmap has format {}, but RGBA_8888 was expected", info.format
		);
	}

	if (out_float_array.size() != (size_t)info.width * (size_t)info.height * 3)
		throw std::invalid_argument("out_float_array");

	void* address_ptr = nullptr;
	result = check_android_bitmap_result(
		AndroidBitmap_lockPixels(env, bitmap, &address_ptr)
	);
	if (!result.has_value())
		return tl::unexpected(result.error());

	if (address_ptr == nullptr) {
		return tl::unexpected("failed to lock bitmap pixels");
	}
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

	return check_android_bitmap_result(AndroidBitmap_unlockPixels(env, bitmap));
}

tl::expected<void, std::string> image_bytes_to_argb_int_array(
	const std::span<const jbyte> image_bytes,
	std::span<jint> out_pixels
) {
	PROFILE_CAMERA_FUNCTION()

	if (image_bytes.size_bytes() != out_pixels.size_bytes()) {
		return tl::unexpected_fmt(
			"image_bytes has {} bytes, but out_pixels has {} bytes!",
			image_bytes.size_bytes(), out_pixels.size_bytes()
		);
	}

	size_t i = 0;
	size_t j = 0;
	for (; i < out_pixels.size(); i++) {
		auto r = image_bytes[j++];
		auto g = image_bytes[j++];
		auto b = image_bytes[j++];
		auto a = image_bytes[j++];
		out_pixels[i] = color_argb(a, r, g, b);
	}

	return {};
}