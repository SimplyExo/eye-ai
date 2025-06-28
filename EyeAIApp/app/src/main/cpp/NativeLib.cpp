#include <android/log.h>
#include <jni.h>
#include <memory>
#include <vector>

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/audio/SpatialAudioEngine.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/utils/DepthColormap.hpp"
#include "EyeAICore/utils/MutexGuard.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "ImageUtils.hpp"
#include "Log.hpp"
#include "NativeJavaScopes.hpp"

// the global variable is using MutexGuard, so they are thread-safe
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static MutexGuard<std::unique_ptr<DepthModel>> depth_model{
	std::unique_ptr<DepthModel>(nullptr)
};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

static std::unique_ptr<SpatialAudioEngine> spatial_audio_engine = nullptr;

// NOLINTBEGIN(readability-identifier-naming,
// bugprone-easily-swappable-parameters)

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_initDepthModel(
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

	auto result = DepthModel::create(
		model_data.to_vector(), gpu_delegate_serialization_dir_string,
		model_token_string, log_warning_callback, log_error_callback
	);
	if (result) {
		depth_model.lock()->swap(*result);
	} else
		LOG_ERROR(
			"[TfLiteRuntime] Failed to create depth model: {}",
			result.error().to_string()
		);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_shutdownDepthModel(
	JNIEnv* /*env*/,
	jobject /*thiz*/
) {
	depth_model.lock()->reset(nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_runDepthModelInference(
	JNIEnv* env,
	jobject /*thiz*/,
	jfloatArray input,
	jfloatArray output
) {
	auto depth_model_scope = depth_model.lock();

	if (*depth_model_scope == nullptr) {
		LOG_ERROR("depth model not initialized!");
		return;
	}

	NativeFloatArrayScope input_array(env, input);
	NativeFloatArrayScope output_array(env, output);

	if (const auto error =
			(*depth_model_scope)->run(input_array, output_array)) {
		LOG_ERROR(
			"[TfLiteRuntime] Failed to run depth model inference: {}",
			error->to_string()
		);
	}
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
		if (const auto error =
				depth_colormap(depth_value_array, colormapped_pixel_array))
			LOG_ERROR("depthColormap failed: {}", error->to_string());
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

	if (const auto error =
			bitmap_to_rgb_chw_float_array(env, bitmap, out_float_array_scope)) {
		LOG_ERROR("bitmapToRgbChwFloatArray failed: {}", error->to_string());
	}
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_bitmapToRgbHwc255FloatArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jobject bitmap,
	jfloatArray out_float_array
) {

	NativeFloatArrayScope out_float_array_scope(env, out_float_array);

	if (const auto error = bitmap_to_rgb_hwc_255_float_array(
			env, bitmap, out_float_array_scope
		)) {
		LOG_ERROR("bitmapToRgbHwc255FloatArray failed: {}", error->to_string());
	}
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

	if (const auto error = image_bytes_to_argb_int_array(
			image_byte_array, out_int_array_scope
		)) {
		LOG_ERROR("imageBytesToArgbIntArray failed: {}", error->to_string());
	}
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_newDepthFrame(
	JNIEnv* /*env*/,
	jobject /*this*/
) {
	set_last_depth_profiling_frame_formatted(
		std::move(get_depth_profiling_frame().finish())
	);
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
	set_last_camera_profiling_frame_formatted(
		std::move(get_camera_profiling_frame().finish())
	);
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