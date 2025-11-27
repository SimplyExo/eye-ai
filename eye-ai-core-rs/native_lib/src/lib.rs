#![allow(non_snake_case)]

use eye_ai_core_rs::{
	CreateDepthModelInfo, CreateYoloModelInfo, DepthModelNpuConfig, FloatTensorBuffer,
	FloatTensorFormat, MetricDepthModel, ObjectTracker, ProfilingFrame, TrackedObject, YoloModel,
	YoloModelNpuConfig,
};
use eye_ai_core_rs_profiling_attribute::profile_function;
use std::{
	ffi::{CStr, CString},
	path::PathBuf,
	sync::{Arc, LazyLock, RwLock},
};

const TFLITE_LIB_FILEPATH: &str = "libtensorflowlite_jni.so";
const TFLITE_GPU_DELEGATE_LIB_FILEPATH: &str = "libtensorflowlite_gpu_jni.so";
const TFLITE_QNN_NPU_DELEGATE_LIB_FILEPATH: &str = "libqnn_delegate_jni.so";

static METRIC_DEPTH_MODEL: LazyLock<RwLock<Option<MetricDepthModel>>> =
	LazyLock::new(|| RwLock::new(None));

static YOLO_MODEL: LazyLock<RwLock<Option<YoloModel>>> = LazyLock::new(|| RwLock::new(None));
static OBJECT_TRACKER: LazyLock<RwLock<Option<ObjectTracker>>> =
	LazyLock::new(|| RwLock::new(None));

static DEPTH_PROFILING_FRAME: LazyLock<ProfilingFrame> =
	LazyLock::new(|| ProfilingFrame::new("Depth"));

static CAMERA_PROFILING_FRAME: LazyLock<ProfilingFrame> =
	LazyLock::new(|| ProfilingFrame::new("Camera"));

static OBJECT_PROFILING_FRAME: LazyLock<ProfilingFrame> =
	LazyLock::new(|| ProfilingFrame::new("Object"));

/// Waits for the RwLock to be free and also waits for the Option to be Some ^= "waits for the model to be loaded"
fn wait_for_model<M, R>(
	profiling_scope_name: &'static str,
	model: &RwLock<Option<M>>,
	f: impl FnOnce(&mut M) -> R,
	profiling_frame: &ProfilingFrame,
) -> R {
	let waiting_scope = profiling_frame.scope(profiling_scope_name);

	let mut model = model.write().unwrap();

	loop {
		if let Some(model) = &mut (*model) {
			drop(waiting_scope); // the 'wait_for_metric_depth_model' scope only shows the waiting time
			return f(model);
		}
	}
}

/// Waits for the RwLock to be free and also waits for the Option to be Some ^= "waits for the model to be loaded"
fn wait_for_metric_depth_model<R>(f: impl FnOnce(&mut MetricDepthModel) -> R) -> R {
	wait_for_model(
		"wait_for_metric_depth_model",
		&METRIC_DEPTH_MODEL,
		f,
		&DEPTH_PROFILING_FRAME,
	)
}

/// Waits for the RwLock to be free and also waits for the Option to be Some ^= "waits for the model to be loaded"
fn wait_for_yolo_model<R>(f: impl FnOnce(&mut YoloModel) -> R) -> R {
	wait_for_model(
		"wait_for_yolo_model",
		&YOLO_MODEL,
		f,
		&OBJECT_PROFILING_FRAME,
	)
}

/*
TODO: This does not get picked up by Kotlin, so for now its in NativeLib c++

if implemented someday:
add "jni = "0.21.1"" to Cargo.toml of native_lib!!!

#[unsafe(no_mangle)]
#[allow(unused)]
pub extern "system" fn Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getByteBufferPtr(
	env: JNIEnv,
	_class: JClass,
	buffer: JByteBuffer,
) -> jlong {
	let ptr = env
		.get_direct_buffer_address(&buffer)
		.expect("not a direct buffer");

	ptr as jlong
}
*/

// Java_com_algorithmic_1alliance_eyeaiapp_NativeLib_getFloatArrayPtr
#[derive(uniffi::Record)]
struct UniffiFloatBufferWrapper {
	/// i64 ^= Long, direct pointer address of an FloatBuffer
	ptr_address: i64,
	/// length of the FloatArray
	length: i32,
}
impl UniffiFloatBufferWrapper {
	fn as_slice_mut(&mut self) -> &mut [f32] {
		unsafe {
			std::slice::from_raw_parts_mut(self.ptr_address as *mut f32, self.length as usize)
		}
	}
}

#[uniffi::export(callback_interface)]
trait LogCallbacks: Send + Sync {
	fn log_info_callback(&self, msg: String);
	fn log_warning_callback(&self, msg: String);
	fn log_error_callback(&self, msg: String);
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
fn initMetricDepthModel(
	relative_depth_model: Vec<u8>,
	gpu_delegate_serialization_dir: String,
	relative_depth_model_token: String,
	enable_npu: bool,
	skel_directory: String,
	logger: Box<dyn LogCallbacks>,
) {
	let metric_depth_model = MetricDepthModel::new(
		CreateDepthModelInfo {
			tflite_lib_filepath: PathBuf::from(TFLITE_LIB_FILEPATH),
			tflite_gpu_delegate_lib_filepath: Some(PathBuf::from(TFLITE_GPU_DELEGATE_LIB_FILEPATH)),
			model_data: relative_depth_model,
			gpu_delegate_serialization_dir: CString::new(gpu_delegate_serialization_dir).unwrap(),
			model_token: CString::new(relative_depth_model_token).unwrap(),
			log_warning_callback: Arc::new(move |msg| {
				logger.log_warning_callback(format!("[TFLITE] {}", msg))
			}),
			// TODO: use logger.log_error_callback instead of eprintln,
			// Problem: error callback needs to be fn(*const std::os::raw::c_char) for interop with c/c++
			log_error_callback: |msg| unsafe {
				match CStr::from_ptr(msg).to_str() {
					Ok(msg) => eprintln!("[ERROR] {}", msg),
					Err(_) => eprintln!(
						"[ERROR] Failed to convert CString to String in order to log tflite error message"
					),
				}
			},
			npu_config: if enable_npu {
				Some(DepthModelNpuConfig {
					tflite_qnn_npu_delegate_lib_filepath: PathBuf::from(
						TFLITE_QNN_NPU_DELEGATE_LIB_FILEPATH,
					),
					skel_library_dir: CString::new(skel_directory).unwrap(),
				})
			} else {
				None
			},
		},
		&DEPTH_PROFILING_FRAME,
	)
	.expect("failed to create metric depth model");

	// TODO: add this back some day if needed, currently problem with moved `logger`
	// logger.log_info_callback("created metric depth model".to_string());

	*METRIC_DEPTH_MODEL.write().unwrap() = Some(metric_depth_model);
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
fn shutdownMetricDepthModel() {
	*METRIC_DEPTH_MODEL.write().unwrap() = None;
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
fn runMetricDepthModelInference(
	mut input: UniffiFloatBufferWrapper,
	mut output: UniffiFloatBufferWrapper,
	logger: Box<dyn LogCallbacks>,
) {
	logger.log_info_callback("running metric depth model".to_string());

	wait_for_metric_depth_model(|metric_depth_model| {
		match metric_depth_model.run(
			&mut FloatTensorBuffer::new(input.as_slice_mut(), FloatTensorFormat::MiDaSImageRgb),
			&mut FloatTensorBuffer::new(output.as_slice_mut(), FloatTensorFormat::MetricDepth),
		) {
			Ok(()) => {}
			Err(e) => {
				logger.log_error_callback(format!(
					"Failed to run metric depth model: {}, output will not be changed",
					e
				));
			}
		}
	})
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
fn getMetricDepthModelInputShape(logger: Box<dyn LogCallbacks>) -> Vec<i32> {
	wait_for_metric_depth_model(
		|metric_depth_model| match metric_depth_model.get_input_shape() {
			Some(input_shape) => input_shape,
			None => {
				logger.log_error_callback(
					"could not get input shape (returning empty vector)".to_string(),
				);
				vec![]
			}
		},
	)
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
fn getMetricDepthModelOutputShape(logger: Box<dyn LogCallbacks>) -> Vec<i32> {
	wait_for_metric_depth_model(
		|metric_depth_model| match metric_depth_model.get_output_shape() {
			Some(output_shape) => output_shape,
			None => {
				logger.log_error_callback(
					"could not get output shape (returning empty vector)".to_string(),
				);
				vec![]
			}
		},
	)
}

#[uniffi::export]
#[profile_function("OBJECT_PROFILING_FRAME")]
fn initYoloRuntime(
	model: Vec<u8>,
	labels: Vec<String>,
	gpu_delegate_serialization_dir: String,
	model_token: String,
	enable_npu: bool,
	skel_directory: String,
	logger: Box<dyn LogCallbacks>,
) {
	let yolo_model = YoloModel::new(
		CreateYoloModelInfo {
			labels: labels.clone(),
			tflite_lib_filepath: PathBuf::from(TFLITE_LIB_FILEPATH),
			tflite_gpu_delegate_lib_filepath: Some(PathBuf::from(TFLITE_GPU_DELEGATE_LIB_FILEPATH)),
			model_data: model,
			gpu_delegate_serialization_dir: CString::new(gpu_delegate_serialization_dir).unwrap(),
			model_token: CString::new(model_token).unwrap(),
			log_warning_callback: Arc::new(move |msg| {
				logger.log_warning_callback(format!("[TFLITE] {}", msg))
			}),
			// TODO: use logger.log_error_callback instead of eprintln,
			// Problem: error callback needs to be fn(*const std::os::raw::c_char) for interop with c/c++
			log_error_callback: |msg| unsafe {
				match CStr::from_ptr(msg).to_str() {
					Ok(msg) => eprintln!("[ERROR] {}", msg),
					Err(_) => eprintln!(
						"[ERROR] Failed to convert CString to String in order to log tflite error message"
					),
				}
			},
			npu_config: if enable_npu {
				Some(YoloModelNpuConfig {
					tflite_qnn_npu_delegate_lib_filepath: PathBuf::from(
						TFLITE_QNN_NPU_DELEGATE_LIB_FILEPATH,
					),
					skel_library_dir: CString::new(skel_directory).unwrap(),
				})
			} else {
				None
			},
		},
		&OBJECT_PROFILING_FRAME,
	)
	.expect("failed to create yolo model");

	*YOLO_MODEL.write().unwrap() = Some(yolo_model);

	let object_tracker = ObjectTracker::new(labels, &OBJECT_PROFILING_FRAME);
	*OBJECT_TRACKER.write().unwrap() = Some(object_tracker);
}

#[derive(uniffi::Record)]
struct UniffiDetectedObject {
	x1: f32,
	y1: f32,
	x2: f32,
	y2: f32,
	cx: f32,
	cy: f32,
	w: f32,
	h: f32,
	cnf: f32,
	cls: i32,
	clsName: String,
	trackingId: i32,
}
impl From<TrackedObject> for UniffiDetectedObject {
	fn from(value: TrackedObject) -> Self {
		let bbox = value.object.bbox;

		Self {
			x1: bbox.x1(),
			y1: bbox.y1(),
			x2: bbox.x2(),
			y2: bbox.y2(),
			cx: bbox.center_x,
			cy: bbox.center_y,
			w: bbox.width,
			h: bbox.height,
			cnf: value.object.confidence,
			cls: value.object.class_id as i32,
			clsName: value.object.class_name,
			trackingId: value.tracking_id,
		}
	}
}

#[uniffi::export]
#[profile_function("OBJECT_PROFILING_FRAME")]
fn runYoloOperation(
	mut input: UniffiFloatBufferWrapper,
	logger: Box<dyn LogCallbacks>,
) -> Vec<UniffiDetectedObject> {
	logger.log_info_callback("running Yolo Operation...".to_string());

	wait_for_yolo_model(|yolo_model| {
		match yolo_model.run(&mut FloatTensorBuffer::new(
			input.as_slice_mut(),
			FloatTensorFormat::ImageRgb255,
		)) {
			Ok(detected_objects) => {
				let tracked_objects = OBJECT_TRACKER
					.write()
					.unwrap()
					.as_mut()
					.expect("OBJECT_TRACKER should have been created when yolo model was created")
					.update(detected_objects);
				tracked_objects
					.into_iter()
					.map(|obj| obj.into())
					.collect::<Vec<_>>()
			}
			Err(e) => {
				logger.log_error_callback(format!(
					"Failed to run yolo model: {}, returning empty float array",
					e
				));
				vec![]
			}
		}
	})
}

#[uniffi::export]
#[profile_function("OBJECT_PROFILING_FRAME")]
fn getYoloInputShape(logger: Box<dyn LogCallbacks>) -> Vec<i32> {
	wait_for_yolo_model(|yolo_model| match yolo_model.get_input_shape() {
		Some(input_shape) => input_shape,
		None => {
			logger.log_error_callback(
				"could not get input shape (returning empty vector)".to_string(),
			);
			vec![]
		}
	})
}

#[uniffi::export]
#[profile_function("OBJECT_PROFILING_FRAME")]
fn getYoloOutputShape(logger: Box<dyn LogCallbacks>) -> Vec<i32> {
	wait_for_yolo_model(|yolo_model| match yolo_model.get_output_shape() {
		Some(output_shape) => output_shape,
		None => {
			logger.log_error_callback(
				"could not get output shape (returning empty vector)".to_string(),
			);
			vec![]
		}
	})
}

#[uniffi::export]
fn formattedDepthFrame() -> String {
	DEPTH_PROFILING_FRAME.formatted_info.read().unwrap().clone()
}

#[uniffi::export]
fn formattedObjectFrame() -> String {
	OBJECT_PROFILING_FRAME
		.formatted_info
		.read()
		.unwrap()
		.clone()
}

#[uniffi::export]
fn formattedCameraFrame() -> String {
	CAMERA_PROFILING_FRAME
		.formatted_info
		.read()
		.unwrap()
		.clone()
}

uniffi::setup_scaffolding!("NativeLib");
