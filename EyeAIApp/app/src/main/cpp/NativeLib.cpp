#include <android/log.h>
#include <jni.h>
#include <memory>
#include <vector>

#include "EyeAICore/DepthEstimation.hpp"
#include "EyeAICore/audio/SpatialAudioEngine.hpp"
#include "EyeAICore/onnx/OnnxRuntime.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/utils/MutexGuard.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "ImageUtils.hpp"
#include "Log.hpp"
#include "NativeJavaScopes.hpp"

// these 2 global variables are using MutexGuard, so they are thread-safe
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static MutexGuard<std::unique_ptr<TfLiteRuntime>>
	depth_estimation_tflite_runtime{std::unique_ptr<TfLiteRuntime>(nullptr)};

static MutexGuard<std::unique_ptr<OnnxRuntime>> depth_estimation_onnx_runtime{
	std::unique_ptr<OnnxRuntime>(nullptr)
};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

static std::unique_ptr<SpatialAudioEngine> spatial_audio_engine = nullptr;

// NOLINTBEGIN(readability-identifier-naming,
// bugprone-easily-swappable-parameters)

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_initDepthTfLiteRuntime(
	JNIEnv* env,
	jobject /*thiz*/,
	jbyteArray model,
	jstring gpu_delegate_serialization_dir,
	jstring model_token
) {

	NativeByteArrayScope model_data(env, model);
	const NativeStringScope gpu_delegate_serialization_dir_string(
		env, gpu_delegate_serialization_dir
	);
	const NativeStringScope model_token_string(env, model_token);

	const auto log_warning_callback = [](std::string msg) {
		LOG_WARN("[TfLiteRuntime] {}", msg);
	};

	const auto log_error_callback = [](std::string msg) {
		LOG_ERROR("[TfLiteRuntime] {}", msg);
	};

	auto result = TfLiteRuntime::create(
		model_data.to_vector(), gpu_delegate_serialization_dir_string,
		model_token_string, log_warning_callback, log_error_callback
	);
	if (result) {
		depth_estimation_tflite_runtime.lock()->swap(*result);
	} else
		LOG_ERROR("[TfLiteRuntime] Failed to create: {}", result.error());
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_shutdownDepthTfLiteRuntime(
	JNIEnv* /*env*/,
	jobject /*thiz*/
) {
	depth_estimation_tflite_runtime.lock()->reset(nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_runDepthTfLiteInference(
	JNIEnv* env,
	jobject /*thiz*/,
	jfloatArray input,
	jfloatArray output,
	jfloat mean_r,
	jfloat mean_g,
	jfloat mean_b,
	jfloat stddev_r,
	jfloat stddev_g,
	jfloat stddev_b
) {
	auto depth_estimation_tflite_runtime_scope =
		depth_estimation_tflite_runtime.lock();

	if (*depth_estimation_tflite_runtime_scope == nullptr) {
		LOG_ERROR("TfLiteRuntime not initialized!");
		return;
	}

	NativeFloatArrayScope input_array(env, input);
	NativeFloatArrayScope output_array(env, output);

	const std::array<float, 3> mean = {mean_r, mean_g, mean_b};
	const std::array<float, 3> stddev = {stddev_r, stddev_g, stddev_b};

	const auto result = run_depth_estimation(
		*(*depth_estimation_tflite_runtime_scope), input_array, output_array,
		mean, stddev
	);
	if (!result.has_value())
		LOG_ERROR(
			"[TfLiteRuntime] Failed to run inference: {}", result.error()
		);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_initDepthOnnxRuntime(
	JNIEnv* env,
	jobject /*thiz*/,
	jbyteArray model
) {
	NativeByteArrayScope model_data(env, model);

	OnnxLogCallbacks log_callbacks{
		.log_info = [](const auto msg) { LOG_INFO("[OnnxRuntime] {}", msg); },
		.log_error = [](const auto msg) { LOG_ERROR("[OnnxRuntime] {}", msg); }
	};

	auto result = OnnxRuntime::create(
		std::as_bytes((std::span<const jbyte>)model_data),
		std::move(log_callbacks)
	);
	if (result)
		depth_estimation_onnx_runtime.lock()->swap(*result);
	else
		LOG_ERROR("[OnnxRuntime] Failed to create: {}", result.error());
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_shutdownDepthOnnxRuntime(
	JNIEnv* /*env*/,
	jobject /*thiz*/
) {
	depth_estimation_onnx_runtime.lock()->reset(nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_runDepthOnnxInference(
	JNIEnv* env,
	jobject /*thiz*/,
	jfloatArray input_data,
	jfloatArray output_data,
	jfloat mean_r,
	jfloat mean_g,
	jfloat mean_b,
	jfloat stddev_r,
	jfloat stddev_g,
	jfloat stddev_b
) {
	auto depth_estimation_onnx_runtime_scope =
		depth_estimation_onnx_runtime.lock();

	if (*depth_estimation_onnx_runtime_scope == nullptr) {
		LOG_ERROR("OnnxRuntime not initialized!");
		return;
	}

	NativeFloatArrayScope input_array(env, input_data);
	NativeFloatArrayScope output_array(env, output_data);

	const std::array<float, 3> mean = {mean_r, mean_g, mean_b};
	const std::array<float, 3> stddev = {stddev_r, stddev_g, stddev_b};

	const auto result = run_depth_estimation(
		*(*depth_estimation_onnx_runtime_scope), input_array, output_array,
		mean, stddev
	);
	if (!result.has_value())
		LOG_ERROR("[OnnxRuntime] Failed to run inference: {}", result.error());
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_depthColormap(
	JNIEnv* env,
	jobject /*thiz*/,
	jfloatArray depth_values,
	jintArray colormapped_pixels
) {
	NativeFloatArrayScope depth_value_array(env, depth_values);
	NativeIntArrayScope colormapped_pixel_array(env, colormapped_pixels);

	if (depth_value_array.size() == colormapped_pixel_array.size()) {
		const auto result =
			depth_colormap(depth_value_array, colormapped_pixel_array);
		if (!result.has_value())
			LOG_ERROR("depthColormap failed: {}", result.error());
	} else {
		LOG_ERROR(
			"depth and colormapped pixel array should have the same length! "
			"({} and {})",
			depth_value_array.size(), colormapped_pixel_array.size()
		);
	}
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_bitmapToRgbChwFloatArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jobject bitmap,
	jfloatArray out_float_array
) {

	NativeFloatArrayScope out_float_array_scope(env, out_float_array);

	const auto result =
		bitmap_to_rgb_chw_float_array(env, bitmap, out_float_array_scope);
	if (!result.has_value())
		LOG_ERROR("bitmapToRgbChwFloatArray failed: {}", result.error());
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_bitmapToRgbHwc255FloatArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jobject bitmap,
	jfloatArray out_float_array
) {

	NativeFloatArrayScope out_float_array_scope(env, out_float_array);

	const auto result =
		bitmap_to_rgb_hwc_255_float_array(env, bitmap, out_float_array_scope);
	if (!result.has_value())
		LOG_ERROR("bitmapToRgbHwc255FloatArray failed: {}", result.error());
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_imageBytesToArgbIntArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jbyteArray image_bytes,
	jintArray out_int_array
) {

	NativeByteArrayScope image_byte_array(env, image_bytes);
	NativeIntArrayScope out_int_array_scope(env, out_int_array);

	const auto result =
		image_bytes_to_argb_int_array(image_byte_array, out_int_array_scope);
	if (!result.has_value())
		LOG_ERROR("imageBytesToArgbIntArray failed: {}", result.error());
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_newDepthFrame(
	JNIEnv* /*env*/,
	jobject /*this*/
) {
	LOG_ON_EXCEPTION(
		get_last_depth_profiling_frame_formatted() =
			get_depth_profiling_frame().finish();
	)
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_formatDepthFrame(
	JNIEnv* env,
	jobject /*this*/
) {
	return env->NewStringUTF(
		get_last_depth_profiling_frame_formatted().c_str()
	);
}
extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_newCameraFrame(
	JNIEnv* /*env*/,
	jobject /*this*/
) {
	LOG_ON_EXCEPTION(
		get_last_camera_profiling_frame_formatted() =
			get_camera_profiling_frame().finish();
	)
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_formatCameraFrame(
	JNIEnv* env,
	jobject /*this*/
) {
	return env->NewStringUTF(
		get_last_camera_profiling_frame_formatted().c_str()
	);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_enableSpatialAudio(
	JNIEnv* /*env*/,
	jobject /*this*/
) {
	AudioLogCallback log_warning_callback = [](std::string_view msg) {
		LOG_WARN("[SpatialAudioEngine] {}", msg);
	};
	AudioLogCallback log_error_callback = [](std::string_view msg) {
		LOG_WARN("[SpatialAudioEngine] {}", msg);
	};

	spatial_audio_engine = std::make_unique<SpatialAudioEngine>(
		log_warning_callback, log_error_callback
	);
	spatial_audio_engine->start();
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_disableSpatialAudio(
	JNIEnv* /*env*/,
	jobject /*this*/
) {
	if (spatial_audio_engine) {
		spatial_audio_engine->stop();
		spatial_audio_engine.reset();
	}
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_updateSpatialAudio(
	JNIEnv* /*env*/,
	jobject /*this*/,
	jfloat amplitude,
	jfloat frequency,
	jfloat position_x,
	jfloat position_y,
	jfloat position_z
) {
	if (spatial_audio_engine == nullptr) {
		LOG_ERROR("SpatialAudioEngine not initialized!");
		return;
	}

	std::vector<OscillatorInfo> oscillators{
		OscillatorInfo(amplitude, frequency, position_x, position_y, position_z)
	};
	LOG_ON_EXCEPTION(spatial_audio_engine->update_oscillators(oscillators);)
}

// NOLINTEND(readability-identifier-naming,
// bugprone-easily-swappable-parameters)