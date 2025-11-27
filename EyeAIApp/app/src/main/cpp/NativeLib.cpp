#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include <EyeAICore/ObjectTracker.hpp>
#include <EyeAICore/audio/AudioMain.hpp>
#include <EyeAICore/audio/SpatialAudio.hpp>
#include <jni.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/MetricDepthModel.hpp"
#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/utils/DepthColormap.hpp"
#include "EyeAICore/utils/MutexGuard.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "ImageUtils.hpp"
#include "Log.hpp"
#include "NativeJavaScopes.hpp"

// the global variables are using MutexGuard, so they are thread-safe
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

// Logging functions for spatial audio
void spatial_audio_log_error_callback(std::string msg);
void spatial_audio_log_info_callback(std::string msg);

namespace {
MutexGuard<std::unique_ptr<SpatialAudio>> spatial_audio{
	std::unique_ptr<SpatialAudio>(nullptr)
};

MutexGuard<SpatialAudioSettings> spatial_audio_settings =
	MutexGuard<SpatialAudioSettings>(SpatialAudioSettings(
		spatial_audio_log_error_callback,
		spatial_audio_log_info_callback
	));

} // namespace
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/// see NativeLib.kt, must be Int!
// NOLINTNEXTLINE(performance-enum-size)
enum class ProfilingFrameType : jint { Depth = 0, Object = 1 };

static ProfilingFrame& get_profiling_frame(ProfilingFrameType type) {
	switch (type) {
	default:
	case ProfilingFrameType::Depth:
		return get_depth_profiling_frame();
	case ProfilingFrameType::Object:
		return get_object_profiling_frame();
	}
}

// NOLINTBEGIN(readability-identifier-naming,
// bugprone-easily-swappable-parameters)

// TODO: move to rust?
extern "C" JNIEXPORT jlong JNICALL Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getByteBufferPtr(
	JNIEnv* env,
	jobject /*_this*/,
	jobject byteBuffer
) {
	return (jlong)env->GetDirectBufferAddress(byteBuffer);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_metricDepthColormap(
	JNIEnv* env,
	jobject /*thiz*/,
	jobject depth_buffer,
	jintArray colormapped_pixels
) {
	std::span<float> depth_span{(float*)env->GetDirectBufferAddress(depth_buffer), (size_t)env->GetDirectBufferCapacity(depth_buffer)};
	NativeIntArrayScope colormapped_pixel_array(env, colormapped_pixels);

	if (depth_span.size() == colormapped_pixel_array.size()) {
		if (const auto error = metric_depth_colormap(
				depth_span, colormapped_pixel_array
			))
			LOG_ERROR("depthColormap failed: {}", error->to_string());
	} else {
		LOG_ERROR(
			"depth and colormapped pixel array should have the same length! "
			"({} and {})",
			depth_span.size(), colormapped_pixel_array.size()
		);
	}
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_bitmapToRgbHwc255FloatArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jobject bitmap,
	jobject out_float_buffer,
	jint profiling_frame_type
) {
	std::span<float> out_float_span{
		(float*)env->GetDirectBufferAddress(out_float_buffer),
		(size_t)env->GetDirectBufferCapacity(out_float_buffer)
	};
	//NativeFloatArrayScope out_float_array_scope(env, out_float_array);

	if (const auto error = bitmap_to_rgb_hwc_255_float_array(
			env, bitmap, out_float_span,
			get_profiling_frame(
				static_cast<ProfilingFrameType>(profiling_frame_type)
			)
		)) {
		LOG_ERROR("bitmapToRgbHwc255FloatArray failed: {}", error->to_string());
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
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_newObjectFrame(
	JNIEnv* /*env*/,
	jobject /*this*/
) {
	set_last_object_profiling_frame_formatted(
		std::move(get_object_profiling_frame().finish())
	);
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_formatObjectFrame(
	JNIEnv* env,
	jobject /*this*/
) {
	return env->NewStringUTF(
		get_last_object_profiling_frame_formatted().c_str()
	);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_setupAudioSettings(
	JNIEnv* env,
	jobject /*this*/,
	jbyteArray coco_labels_audio,
	jbyteArray coco_labels_data
) {
	LOG_INFO("[SpatialAudio] Setting up AudioSettings ... ");
	jbyte* coco_labels_audio_ptr =
		env->GetByteArrayElements(coco_labels_audio, nullptr);
	if (coco_labels_audio_ptr == nullptr) {
		LOG_ERROR("[SpatialAudio] Failed to get coco_labels_audio elements.");
		return;
	}

	jsize coco_labels_audio_size = env->GetArrayLength(coco_labels_audio);

	std::vector<std::byte> coco_labels_audio_vector(coco_labels_audio_size);
	std::memcpy(
		coco_labels_audio_vector.data(), coco_labels_audio_ptr,
		coco_labels_audio_size
	);

	jbyte* coco_labels_data_ptr =
		env->GetByteArrayElements(coco_labels_data, nullptr);
	if (coco_labels_data_ptr == nullptr) {
		LOG_ERROR("[SpatialAudio] Failed to get coco_labels_data elements.");
		env->ReleaseByteArrayElements(
			coco_labels_audio, coco_labels_audio_ptr, JNI_ABORT
		); // Freigeben bei Fehler
		return;
	}
	jsize coco_labels_data_size = env->GetArrayLength(coco_labels_data);

	std::vector<std::byte> coco_labels_data_vector(coco_labels_data_size);
	std::memcpy(
		coco_labels_data_vector.data(), coco_labels_data_ptr,
		coco_labels_data_size
	);
	auto audio_setting_scope = spatial_audio_settings.lock();

	audio_setting_scope->coco_labels_audio.clear();
	audio_setting_scope->coco_labels_audio =
		std::move(coco_labels_audio_vector);
	audio_setting_scope->coco_labels_data.clear();
	audio_setting_scope->coco_labels_data = std::move(coco_labels_data_vector);

	env->ReleaseByteArrayElements(
		coco_labels_audio, coco_labels_audio_ptr, JNI_ABORT
	);
	env->ReleaseByteArrayElements(
		coco_labels_data, coco_labels_data_ptr, JNI_ABORT
	);
	LOG_INFO("[SpatialAudio] Set up AudioSettings ... ");
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_setAudioSettings(
	JNIEnv* env,
	jobject /*this*/,
	jint frequency,
	jint incidence
) {
	LOG_INFO("[SpatialAudio] Updating audio settings...");

	auto audio_setting_scope = spatial_audio_settings.lock();

	audio_setting_scope->FREQUENCY = (float)frequency;
	audio_setting_scope->BUFFER_DURATION = ((float)1) / incidence;
}

void spatial_audio_log_error_callback(std::string msg) {
	LOG_ERROR("[SpatialAudio] {}", msg);
};

void spatial_audio_log_info_callback(std::string msg) {
	LOG_INFO("[SpatialAudio] {}", msg);
};

static SpatialAudio& get_or_create_spatial_audio() {
	auto spatial_audio_scope = spatial_audio.lock();
	auto audio_setting_scope = spatial_audio_settings.lock();

	if (*spatial_audio_scope == nullptr) {
		LOG_INFO("[SpatialAudio] Initializing SpatialAudio instance...");
		*spatial_audio_scope =
			std::make_unique<SpatialAudio>(*audio_setting_scope);
	}

	return *(*spatial_audio_scope);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_setDepthAudioPaused(
	JNIEnv* env,
	jobject /*this*/,
	jboolean paused
) {
	LOG_INFO("[SpatialAudio] Setting depth audio playback. Paused: {}", paused);

	auto audio_setting_scope = spatial_audio_settings.lock();

	audio_setting_scope->depth_audio_paused = paused;
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_setObjectAudioPaused(
	JNIEnv* env,
	jobject /*this*/,
	jboolean paused
) {
	LOG_INFO(
		"[SpatialAudio] Setting object audio playback. Paused: {}", paused
	);

	auto audio_setting_scope = spatial_audio_settings.lock();

	audio_setting_scope->object_audio_paused = paused;
}

// NOTE: since tracked objects are in rust/kotlin, while we port audio to rust, objects is always empty, function will be removed by then
extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_sendAIData(
	JNIEnv* env,
	jobject /*this*/,
	jobject depth_data_buffer
) {
	std::span<float> depth_data_span{
		(float*)env->GetDirectBufferAddress(depth_data_buffer),
		(size_t)env->GetDirectBufferCapacity(depth_data_buffer)
	};

	assert(depth_estimation_data.size() == (256 * 256));
	// EMPTY!
	std::vector<ObjectTracker::TrackedBoundingBox> object_detection_data{};
	get_or_create_spatial_audio().getAIData(
		static_cast<std::span<float, 256 * 256>>(depth_data_span),
		object_detection_data
	);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getProcessingStatus(
	JNIEnv* env,
	jobject /*this*/
) {
	return get_or_create_spatial_audio().getProcessingStatus();
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_destroySpatialAudio(
	JNIEnv* /*env*/,
	jobject /*this*/
) {
	auto spatial_audio_scope = spatial_audio.lock();
	std::this_thread::sleep_for(std::chrono::seconds(3));
	if (*spatial_audio_scope != nullptr) {
		LOG_INFO("[SpatialAudio] Destroying SpatialAudio instance...");
		spatial_audio_scope->reset(nullptr);
		LOG_INFO("[SpatialAudio] SpatialAudio destroyed!");
	} else {
		LOG_INFO("[SpatialAudio] SpatialAudio already destroyed!");
	}
}

// NOLINTEND(readability-identifier-naming,
// bugprone-easily-swappable-parameters)