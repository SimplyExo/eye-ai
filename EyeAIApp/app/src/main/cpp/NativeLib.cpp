#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include <EyeAICore/ObjectTracker.hpp>
#include <EyeAICore/audio/AudioMain.hpp>
#include <EyeAICore/audio/SpatialAudio.hpp>
#include <jni.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/MetricDepthModel.hpp"
#include "EyeAICore/Rel2AbsDepthModel.hpp"
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
MutexGuard<std::unique_ptr<MetricDepthModel>> metric_depth_model{
	std::unique_ptr<MetricDepthModel>(nullptr)
};
MutexGuard<std::unique_ptr<SpatialAudio>> spatial_audio{
	std::unique_ptr<SpatialAudio>(nullptr)
};

MutexGuard<std::unique_ptr<ObjectTracker>> object_tracker;

MutexGuard<YoloModel> yolo_instance;

MutexGuard<std::vector<ObjectTracker::TrackedBoundingBox>> last_tracked_objects;

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

extern "C" JNIEXPORT jboolean JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_initYoloRuntime(
	JNIEnv* env,
	jobject /*thiz*/,
	jbyteArray model,
	jobjectArray labels,
	jstring gpu_delegate_serialization_dir,
	jstring model_token,
	jboolean enable_npu
) {
	NativeByteArrayScope model_data(env, model);
	const NativeStringScope gpu_delegate_serialization_dir_string(
		env, gpu_delegate_serialization_dir
	);
	const NativeStringScope model_token_string(env, model_token);

	const auto log_warning_callback = [](std::string msg) {
		LOG_WARN("[YoloRuntime] {}", msg);
	};

	const auto log_error_callback = [](std::string msg) {
		LOG_ERROR("[YoloRuntime] {}", msg);
	};

	// Labels laden
	jsize len = env->GetArrayLength(labels);
	std::vector<std::string> labels_vector = {};

	for (jsize i = 0; i < len; i++) {
		jstring str = (jstring)env->GetObjectArrayElement(labels, i);

		const char* cstr = env->GetStringUTFChars(str, nullptr);
		labels_vector.push_back(cstr);
		env->ReleaseStringUTFChars(str, cstr);

		env->DeleteLocalRef(str);
	}

	*object_tracker.lock() = std::make_unique<ObjectTracker>(labels_vector);

	auto result = yolo_instance.lock()->create(
		model_data.to_vector(), labels_vector,
		gpu_delegate_serialization_dir_string, model_token_string,
		log_warning_callback, log_error_callback, enable_npu
	);

	if (!result.has_value()) {
		LOG_ERROR(
			"[YoloRuntime] Could not create YoloModel: {}", result.error()
		);
		return false;
	}

	LOG_INFO("[YoloRuntime] Runtime erstellt!");
	return true;
}

static jstring convertTrackedBoundingBoxesToJsonString(
	JNIEnv* env,
	const std::vector<ObjectTracker::TrackedBoundingBox>& tracked_boxes
) {
	nlohmann::json j;

	for (size_t i = 0; i < tracked_boxes.size(); ++i) {
		const ObjectTracker::TrackedBoundingBox& tracked_bounding_box =
			tracked_boxes[i];
		const YoloModel::BoundingBox& bbox = tracked_bounding_box.bounding_box;

		j["bounding_boxes"][i]["clsName"] = bbox.cls_name;
		j["bounding_boxes"][i]["cx"] = bbox.cx;
		j["bounding_boxes"][i]["cy"] = bbox.cy;
		j["bounding_boxes"][i]["w"] = bbox.w;
		j["bounding_boxes"][i]["h"] = bbox.h;
		j["bounding_boxes"][i]["x1"] = bbox.x1;
		j["bounding_boxes"][i]["y1"] = bbox.y1;
		j["bounding_boxes"][i]["x2"] = bbox.x2;
		j["bounding_boxes"][i]["y2"] = bbox.y2;
		j["bounding_boxes"][i]["cls"] = bbox.cls;
		j["bounding_boxes"][i]["cnf"] = bbox.cnf;
		j["bounding_boxes"][i]["trackingId"] = tracked_bounding_box.tracking_id;
	}

	try {
		std::string json_string = j.dump();
		return env->NewStringUTF(json_string.c_str());
	} catch (const std::exception& e) {
		LOG_ERROR(
			"Failed to serialize json string of tracked objects: {}", e.what()
		);
		return env->NewStringUTF("{ \"bounding_boxes\": [] }");
	}
}

extern "C" JNIEXPORT jintArray
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getYoloOutputShape(
	JNIEnv* env,
	jobject /* this */
) {
	auto shape = yolo_instance.lock()->get_output_shape();
	jsize length = static_cast<jsize>(shape.size());

	jintArray array = env->NewIntArray(length);
	if (array == nullptr) {
		// Fehlerbehandlung: Speicher konnte nicht alloziert werden
		return nullptr;
	}

	env->SetIntArrayRegion(array, 0, length, shape.data());

	return array;
}

extern "C" JNIEXPORT jintArray
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getYoloInputShape(
	JNIEnv* env,
	jobject /* this */
) {
	auto shape = yolo_instance.lock()->get_input_shape();
	jsize length = static_cast<jsize>(shape.size());

	jintArray array = env->NewIntArray(length);
	if (array == nullptr) {
		// Fehlerbehandlung: Speicher konnte nicht alloziert werden
		return nullptr;
	}

	env->SetIntArrayRegion(array, 0, length, shape.data());

	return array;
}

extern "C" JNIEXPORT jstring
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_runYoloOperation(
	JNIEnv* env,
	jobject /* this */,
	jfloatArray input
) {
	NativeFloatArrayScope input_scope(env, input);

	FloatTensorBuffer<FloatTensorFormat::ImageRGB255> input_tensor{
		std::span<float>(input_scope)
	};

	const auto result = yolo_instance.lock()->run(input_tensor);
	if (result) {
		const auto tracked_objects = (*object_tracker.lock())->update(*result);
		*last_tracked_objects.lock() = tracked_objects;
		return convertTrackedBoundingBoxesToJsonString(env, tracked_objects);
	} else {
		LOG_ERROR("YoloModel failed to run: {}", result.error());
		return convertTrackedBoundingBoxesToJsonString(
			env, std::vector<ObjectTracker::TrackedBoundingBox>{}
		);
	}
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_initMetricDepthModel(
	JNIEnv* env,
	jobject /*thiz*/,
	jbyteArray relative_depth_model,
	jbyteArray rel2abs_depth_model,
	jstring gpu_delegate_serialization_dir,
	jstring relative_depth_model_token,
	jstring rel2abs_depth_model_token,
	jboolean enable_npu
) {
	const NativeStringScope gpu_delegate_serialization_dir_string(
		env, gpu_delegate_serialization_dir
	);

	NativeByteArrayScope relative_depth_model_data(env, relative_depth_model);
	const NativeStringScope relative_depth_model_token_string(
		env, relative_depth_model_token
	);

	NativeByteArrayScope rel2abs_depth_model_data(env, rel2abs_depth_model);
	const NativeStringScope rel2abs_depth_model_token_string(
		env, rel2abs_depth_model_token
	);

	const auto log_warning_callback = [](std::string msg) {
		LOG_WARN("[TfLiteRuntime] {}", msg);
	};

	const auto log_error_callback = [](std::string msg) {
		LOG_ERROR("[TfLiteRuntime] {}", msg);
	};

	auto result = MetricDepthModel::create(
		relative_depth_model_data.to_vector(),
		rel2abs_depth_model_data.to_vector(),
		gpu_delegate_serialization_dir_string,
		relative_depth_model_token_string, rel2abs_depth_model_token_string,
		log_warning_callback, log_error_callback, enable_npu
	);
	if (result) {
		metric_depth_model.lock()->swap(*result);
	} else
		LOG_ERROR(
			"[TfLiteRuntime] Failed to create depth model: {}",
			result.error().to_string()
		);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_shutdownMetricDepthModel(
	JNIEnv* /*env*/,
	jobject /*thiz*/
) {
	metric_depth_model.lock()->reset(nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_runMetricDepthModelInference(
	JNIEnv* env,
	jobject /*thiz*/,
	jfloatArray input,
	jfloatArray output
) {
	auto depth_model_scope = metric_depth_model.lock();

	if (*depth_model_scope == nullptr) {
		LOG_ERROR("depth model not initialized!");
		return;
	}

	NativeFloatArrayScope input_array(env, input);
	NativeFloatArrayScope output_array(env, output);

	FloatTensorBuffer<FloatTensorFormat::ImageRGB255> input_tensor{
		std::span<float>(input_array)
	};

	auto result = (*depth_model_scope)->run(input_tensor);

	if (result) {
		auto depth_output = result->data();
		if (depth_output.size() == output_array.size()) {
			std::ranges::copy(depth_output, output_array.begin());
		} else {
			LOG_ERROR(
				"DepthModel: invalid output float array size of {} does not "
				"match the expected size of {} from the model",
				output_array.size(), depth_output.size()
			);
		}
	} else {
		LOG_ERROR(
			"[TfLiteRuntime] Failed to run depth model inference: {}",
			result.error().to_string()
		);
	}
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getMetricDepthModelInputShape(
	JNIEnv* env,
	jobject /*thiz*/
) {
	std::span<const int> input_shape =
		(*metric_depth_model.lock())->get_input_shape();

	return create_jni_int_array(env, input_shape);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getMetricDepthModelOutputShape(
	JNIEnv* env,
	jobject /*thiz*/
) {
	std::span<const int> output_shape =
		(*metric_depth_model.lock())->get_output_shape();

	return create_jni_int_array(env, output_shape);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_metricDepthColormap(
	JNIEnv* env,
	jobject /*thiz*/,
	jfloatArray depth_values,
	jintArray colormapped_pixels
) {
	NativeFloatArrayScope depth_value_array(env, depth_values);
	NativeIntArrayScope colormapped_pixel_array(env, colormapped_pixels);

	if (depth_value_array.size() == colormapped_pixel_array.size()) {
		if (const auto error = metric_depth_colormap(
				depth_value_array, colormapped_pixel_array
			))
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
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_bitmapToRgbHwc255FloatArray(
	JNIEnv* env,
	jobject /*thiz*/,
	jobject bitmap,
	jfloatArray out_float_array,
	jint profiling_frame_type
) {
	NativeFloatArrayScope out_float_array_scope(env, out_float_array);

	if (const auto error = bitmap_to_rgb_hwc_255_float_array(
			env, bitmap, out_float_array_scope,
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

	audio_setting_scope->coco_labels_audio =
		std::move(coco_labels_audio_vector);
	audio_setting_scope->coco_labels_data = std::move(coco_labels_data_vector);

	env->ReleaseByteArrayElements(
		coco_labels_audio, coco_labels_audio_ptr, JNI_ABORT
	);
	env->ReleaseByteArrayElements(
		coco_labels_data, coco_labels_data_ptr, JNI_ABORT
	);
}

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_setAudioSettings(
	JNIEnv* env,
	jobject /*this*/,
	jint number_of_sources,
	jfloat frequency
) {
	LOG_INFO("[SpatialAudio] Updating audio settings...");

	auto audio_setting_scope = spatial_audio_settings.lock();

	audio_setting_scope->FREQUENCY = frequency;
	audio_setting_scope->NUMBER_OF_SOURCES = number_of_sources;
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

extern "C" JNIEXPORT void JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_sendAIData(
	JNIEnv* env,
	jobject /*this*/,
	jfloatArray depth_data_array
) {
	jfloat* rawArray = env->GetFloatArrayElements(depth_data_array, nullptr);

	NativeFloatArrayScope depth_estimation_data(env, depth_data_array);

	assert(depth_estimation_data.size() == (256 * 256));
	std::vector<ObjectTracker::TrackedBoundingBox> object_detection_data =
		*last_tracked_objects.lock();
	get_or_create_spatial_audio().getAIData(
		static_cast<std::span<float, 256 * 256>>(depth_estimation_data),
		object_detection_data
	);

	// Speicher freigeben
	env->ReleaseFloatArrayElements(depth_data_array, rawArray, JNI_ABORT);
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