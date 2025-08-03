#include <jni.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/utils/DepthColormap.hpp"
#include "EyeAICore/utils/MutexGuard.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "ImageUtils.hpp"
#include "Log.hpp"
#include "NativeJavaScopes.hpp"

// the global variables are using MutexGuard, so they are thread-safe
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

namespace {
MutexGuard<std::unique_ptr<DepthModel>> depth_model{
	std::unique_ptr<DepthModel>(nullptr)
};

MutexGuard<YoloModel> yolo_instance;
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
	jstring model_token
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

	auto result = yolo_instance.lock()->create(
		model_data.to_vector(), labels_vector,
		gpu_delegate_serialization_dir_string, model_token_string,
		log_warning_callback, log_error_callback
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

static jstring convertToJsonBoundingBoxString(
	JNIEnv* env,
	const std::vector<YoloModel::BoundingBox>& boxes
) {
	nlohmann::json j;

	for (size_t i = 0; i < boxes.size(); ++i) {
		const YoloModel::BoundingBox& b = boxes[i];

		j["bounding_boxes"][i]["clsName"] = b.cls_name;
		j["bounding_boxes"][i]["cx"] = b.cx;
		j["bounding_boxes"][i]["cy"] = b.cy;
		j["bounding_boxes"][i]["w"] = b.w;
		j["bounding_boxes"][i]["h"] = b.h;
		j["bounding_boxes"][i]["x1"] = b.x1;
		j["bounding_boxes"][i]["y1"] = b.y1;
		j["bounding_boxes"][i]["x2"] = b.x2;
		j["bounding_boxes"][i]["y2"] = b.y2;
		j["bounding_boxes"][i]["cls"] = b.cls;
		j["bounding_boxes"][i]["cnf"] = b.cnf;
	}

	return env->NewStringUTF(j.dump().c_str());
}

extern "C" JNIEXPORT jintArray
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getOutputShape(
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
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getInputShape(
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

	// Get input array length safely
	jsize input_length = env->GetArrayLength(input);

	// Allocate buffer for input
	std::vector<float> converted_input(input_length);
	env->GetFloatArrayRegion(input, 0, input_length, converted_input.data());

	auto yolo_instance_scope = yolo_instance.lock();
	// Allocate output buffer (make sure size matches model output)
	std::vector<float> object_recognition_output(
		yolo_instance_scope->num_channel * yolo_instance_scope->num_elements
	); // Replace with actual expected output size

	// Run inference
	const auto exec =
		yolo_instance_scope->run(converted_input, object_recognition_output);

	// Find best boxes
	auto boxes = yolo_instance_scope->best_box(object_recognition_output);

	return convertToJsonBoundingBoxString(env, boxes);
}

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

extern "C" JNIEXPORT jintArray JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getDepthModelInputShape(
	JNIEnv* env,
	jobject /*thiz*/
) {
	std::span<const int> input_shape = (*depth_model.lock())->get_input_shape();

	return create_jni_int_array(env, input_shape);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getDepthModelOutputShape(
	JNIEnv* env,
	jobject /*thiz*/
) {
	std::span<const int> output_shape =
		(*depth_model.lock())->get_output_shape();

	return create_jni_int_array(env, output_shape);
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

// NOLINTEND(readability-identifier-naming,
// bugprone-easily-swappable-parameters)