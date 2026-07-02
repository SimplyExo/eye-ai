#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <memory>
#include <span>
#include <format>

template<typename... Args>
static void formatted_log(int priority, const char* format, Args... args) {
	constexpr static const char* TAG = "Native Lib";

#if __ANDROID_API__ >= 30
	if (!__android_log_is_loggable(priority, TAG, ANDROID_LOG_INFO))
		return;
#endif

	try {
		const std::string formatted =
		std::vformat(format, std::make_format_args(args...));

		__android_log_write(priority, TAG, formatted.c_str());
	} catch (const std::format_error& e) {
		__android_log_write(ANDROID_LOG_ERROR, TAG, e.what());
	}
}

#define LOG_INFO(...) formatted_log(ANDROID_LOG_INFO, __VA_ARGS__)
#define LOG_ERROR(...) formatted_log(ANDROID_LOG_ERROR, __VA_ARGS__)


constexpr static uint8_t red_channel_from_argb_color(int color) {
	return (color >> 16) & 255;
}
constexpr static uint8_t green_channel_from_argb_color(int color) {
	return (color >> 8) & 255;
}
constexpr static uint8_t blue_channel_from_argb_color(int color) {
	return color & 255;
}

struct [[nodiscard]] BitmapError {
	std::string error_msg;

	[[nodiscard]] std::string to_string() const { return error_msg; }

	template<typename... Args>
	[[nodiscard]] static BitmapError
	fmt(const std::format_string<Args...> fmt, Args... args) {
		return BitmapError(
			std::vformat(fmt.get(), std::make_format_args(args...))
		);
	}
};

[[nodiscard]] static std::optional<BitmapError> check_android_bitmap_result(int result) {
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

/// converts pixel from bitmap into float array with (height, width, channel)
/// shape and 3 rgb-channels each in the range of 0.0f to 255.0f
/// often the right format for use with tflite models
[[nodiscard]] static std::optional<BitmapError> bitmap_to_rgb_hwc_255_float_array(
	JNIEnv* env,
	jobject bitmap,
	std::span<float> out_float_array
) {
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

	if (out_float_array.size() != static_cast<size_t>(info.width) * static_cast<size_t>(info.height) * 3)
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
		static_cast<size_t>(info.width) * static_cast<size_t>(info.height)
	);
	// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

	size_t i = 0;
	size_t j = 0;
	for (; i < static_cast<size_t>(info.width) * (size_t)info.height; i++) {
		const int pixel_color = pixel_ptr[i];
		out_float_array[j++] = static_cast<float>(red_channel_from_argb_color(pixel_color));
		out_float_array[j++] =
			static_cast<float>(green_channel_from_argb_color(pixel_color));
		out_float_array[j++] = static_cast<float>(blue_channel_from_argb_color(pixel_color));
	}

	return check_android_bitmap_result(AndroidBitmap_unlockPixels(env, bitmap));
}

// NOLINTBEGIN(readability-identifier-naming,
// bugprone-easily-swappable-parameters)

// TODO: move to rust?
extern "C" JNIEXPORT jlong JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getByteBufferPtr(
	JNIEnv* env,
	jobject /*_this*/,
	jobject byteBuffer
) {
	return (jlong)env->GetDirectBufferAddress(byteBuffer);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_bitmapToRgbHwc255FloatArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jobject bitmap,
	jobject out_float_buffer
) {
	std::span<float> out_float_span{
		(float*)env->GetDirectBufferAddress(out_float_buffer),
		(size_t)env->GetDirectBufferCapacity(out_float_buffer)
	};
	// NativeFloatArrayScope out_float_array_scope(env, out_float_array);

	if (const auto error = bitmap_to_rgb_hwc_255_float_array(
			env, bitmap, out_float_span
		)) {
		LOG_ERROR("bitmapToRgbHwc255FloatArray failed: {}", error->to_string());
	}
}

// NOLINTEND(readability-identifier-naming,
// bugprone-easily-swappable-parameters)