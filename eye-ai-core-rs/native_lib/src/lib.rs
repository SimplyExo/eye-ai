#![allow(non_snake_case)]

use eye_ai_core_rs::{
	BoundingBox, CreateDepthModelInfo, CreateYoloModelInfo, DepthModelNpuConfig, DetectedObject,
	FloatTensorBuffer, FloatTensorFormat, MetricDepthModel, ObjectTracker, ProfilingFrame,
	TrackedObject, YoloModel, YoloModelNpuConfig,
	audio::{
		SpatialAudio, SpatialAudioContent, SpatialAudioSettings, read_audio_file,
		read_object_label_data,
	},
	inferno_colormap,
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
static LAST_FORMATTED_DEPTH_PROFILING_INFO: LazyLock<RwLock<String>> =
	LazyLock::new(|| RwLock::new(String::new()));

static OBJECT_PROFILING_FRAME: LazyLock<ProfilingFrame> =
	LazyLock::new(|| ProfilingFrame::new("Object"));
static LAST_FORMATTED_OBJECT_PROFILING_INFO: LazyLock<RwLock<String>> =
	LazyLock::new(|| RwLock::new(String::new()));

static SPATIAL_AUDIO: LazyLock<RwLock<Option<SpatialAudio>>> = LazyLock::new(|| RwLock::new(None));
static SPATIAL_AUDIO_SETTINGS: LazyLock<Arc<RwLock<SpatialAudioSettings>>> =
	LazyLock::new(|| Arc::new(RwLock::new(SpatialAudioSettings::default())));
static SPATIAL_AUDIO_CONTENT: LazyLock<RwLock<Option<Arc<SpatialAudioContent>>>> =
	LazyLock::new(|| RwLock::new(None));
// TODO: Display audio profiling frame info in EyeAIApp
static AUDIO_PROFILING_FRAME: LazyLock<Arc<ProfilingFrame>> =
	LazyLock::new(|| Arc::new(ProfilingFrame::new("Audio")));

/// Waits for the RwLock to be free and also waits for the Option to be Some ^= "waits for the model to be loaded"
fn wait_for_model<M, R>(
	profiling_scope_name: &'static str,
	model: &RwLock<Option<M>>,
	f: impl FnOnce(&mut M) -> R,
	profiling_frame: &ProfilingFrame,
) -> R {
	let waiting_scope = profiling_frame.scope(profiling_scope_name);

	loop {
		if let Some(model) = &mut (*model.write().unwrap()) {
			drop(waiting_scope); // the 'waiting_scope' scope only shows the waiting time
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

fn try_change_spatial_audio<R>(f: impl FnOnce(&mut SpatialAudio) -> R) -> Option<R> {
	let waiting_scope = AUDIO_PROFILING_FRAME.scope("try_mutate_spatial_audio");

	// first, wait for the spatial audio content to be loaded
	if let Some(content) = &(*SPATIAL_AUDIO_CONTENT.read().unwrap()) {
		// then wait for the spatial audio lock (also: create it, if it does not exist yet)
		let spatial_audio = &mut (*SPATIAL_AUDIO.write().unwrap());
		let spatial_audio = spatial_audio.get_or_insert_with(|| {
			SpatialAudio::new(
				SPATIAL_AUDIO_SETTINGS.clone(),
				content.clone(),
				AUDIO_PROFILING_FRAME.clone(),
			)
			.unwrap()
		});
		drop(waiting_scope);
		Some(f(spatial_audio))
	} else {
		None
	}
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
pub struct UniffiFloatBufferWrapper {
	/// i64 ^= Long, direct pointer address of an FloatBuffer
	pub ptr_address: i64,
	/// length of the FloatArray
	pub length: i32,
}
impl UniffiFloatBufferWrapper {
	fn as_slice_mut(&mut self) -> &mut [f32] {
		unsafe {
			std::slice::from_raw_parts_mut(self.ptr_address as *mut f32, self.length as usize)
		}
	}
}

#[derive(uniffi::Record)]
pub struct UniffiIntBufferWrapper {
	/// i64 ^= Long, direct pointer address of an IntBuffer
	pub ptr_address: i64,
	/// length of the IntArray
	pub length: i32,
}
impl UniffiIntBufferWrapper {
	fn as_slice_mut(&mut self) -> &mut [i32] {
		unsafe {
			std::slice::from_raw_parts_mut(self.ptr_address as *mut i32, self.length as usize)
		}
	}
}

#[uniffi::export(callback_interface)]
pub trait LogCallbacks: Send + Sync {
	fn log_info_callback(&self, msg: String);
	fn log_warning_callback(&self, msg: String);
	fn log_error_callback(&self, msg: String);
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
pub fn initMetricDepthModel(
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
pub fn shutdownMetricDepthModel() {
	*METRIC_DEPTH_MODEL.write().unwrap() = None;
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
pub fn runMetricDepthModelInference(
	mut input: UniffiFloatBufferWrapper,
	mut output: UniffiFloatBufferWrapper,
	logger: Box<dyn LogCallbacks>,
) {
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
pub fn getMetricDepthModelInputShape(logger: Box<dyn LogCallbacks>) -> Vec<i32> {
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
pub fn getMetricDepthModelOutputShape(logger: Box<dyn LogCallbacks>) -> Vec<i32> {
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
#[profile_function("DEPTH_PROFILING_FRAME")]
pub fn metricDepthColormap(
	mut depth_buffer: UniffiFloatBufferWrapper,
	mut colormapped_pixels: UniffiIntBufferWrapper,
	logger: Box<dyn LogCallbacks>,
) {
	const MAX_METRIC_DISTANCE: f32 = 5.0;

	let depth_buffer = depth_buffer.as_slice_mut();
	let colormapped_pixels = colormapped_pixels.as_slice_mut();
	if depth_buffer.len() == colormapped_pixels.len() {
		for (i, depth_value) in depth_buffer.iter().enumerate() {
			colormapped_pixels[i] = inferno_colormap(depth_value / MAX_METRIC_DISTANCE);
		}
	} else {
		logger.log_error_callback(format!(
			"depth_buffer and colormapped_pixels have different sizes: {} and {} (not changing colormapped_pixels!)",
			depth_buffer.len(),
			colormapped_pixels.len()
		));
	}
}

#[uniffi::export]
#[profile_function("OBJECT_PROFILING_FRAME")]
pub fn initYoloRuntime(
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
pub struct UniffiDetectedObject {
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
impl From<UniffiDetectedObject> for TrackedObject {
	fn from(value: UniffiDetectedObject) -> Self {
		Self {
			object: DetectedObject::new(
				value.clsName,
				value.cls as usize,
				value.cnf,
				BoundingBox::new(value.cx, value.cy, value.w, value.h),
			),
			tracking_id: value.trackingId,
		}
	}
}

#[uniffi::export]
#[profile_function("OBJECT_PROFILING_FRAME")]
pub fn runYoloOperation(
	mut input: UniffiFloatBufferWrapper,
	logger: Box<dyn LogCallbacks>,
) -> Vec<UniffiDetectedObject> {
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
pub fn getYoloInputShape(logger: Box<dyn LogCallbacks>) -> Vec<i32> {
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
pub fn getYoloOutputShape(logger: Box<dyn LogCallbacks>) -> Vec<i32> {
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
pub fn newDepthFrame() {
	if let Some(new_formatted_info) = DEPTH_PROFILING_FRAME.finish() {
		*LAST_FORMATTED_DEPTH_PROFILING_INFO.write().unwrap() = new_formatted_info;
	}
}
#[uniffi::export]
pub fn formattedDepthFrame() -> String {
	LAST_FORMATTED_DEPTH_PROFILING_INFO.read().unwrap().clone()
}

#[uniffi::export]
pub fn newObjectFrame() {
	if let Some(new_formatted_info) = OBJECT_PROFILING_FRAME.finish() {
		*LAST_FORMATTED_OBJECT_PROFILING_INFO.write().unwrap() = new_formatted_info;
	}
}
#[uniffi::export]
pub fn formattedObjectFrame() -> String {
	LAST_FORMATTED_OBJECT_PROFILING_INFO.read().unwrap().clone()
}

#[uniffi::export]
pub fn setupAudioContent(
	coco_labels_audio_file_content: Vec<u8>,
	coco_labels_json_content: String,
	logger: Box<dyn LogCallbacks>,
) {
	logger.log_info_callback("loading audio content...".to_string());

	let coco_audio_file = match read_audio_file(
		coco_labels_audio_file_content.into_boxed_slice(),
		&AUDIO_PROFILING_FRAME,
	) {
		Ok(coco_audio_file) => coco_audio_file,
		Err(e) => {
			logger.log_error_callback(format!("Failed to read coco labels audio file: {e}"));
			return;
		}
	};

	let coco_labels_data =
		match read_object_label_data(&coco_labels_json_content, &AUDIO_PROFILING_FRAME) {
			Ok(coco_labels_data) => coco_labels_data,
			Err(e) => {
				logger.log_error_callback(format!("Failed to read object label json data: {e}"));
				return;
			}
		};

	let content = Arc::new(SpatialAudioContent::new(coco_audio_file, coco_labels_data));

	SPATIAL_AUDIO_CONTENT
		.write()
		.unwrap()
		.replace(content.clone());
}

fn createSpatialAudio_impl(logger: &dyn LogCallbacks) {
	logger.log_info_callback("creating spatial audio...".to_string());

	let Some(spatial_audio_content) = &(*SPATIAL_AUDIO_CONTENT.read().unwrap()) else {
		logger.log_error_callback("SPATIAL_AUDIO_CONTENT needs to be setup by calling setupAudioContent before calling createSpatialAudio".to_string());
		return;
	};

	let spatial_audio = SpatialAudio::new(
		SPATIAL_AUDIO_SETTINGS.clone(),
		spatial_audio_content.clone(),
		AUDIO_PROFILING_FRAME.clone(),
	)
	.expect("failed to create spatial audio");

	SPATIAL_AUDIO.write().unwrap().replace(spatial_audio);
}

/// This requires the SPATIAL_AUDIO_CONTENT to be set by calling setupAudioContent before this function
#[uniffi::export]
pub fn createSpatialAudio(logger: Box<dyn LogCallbacks>) {
	createSpatialAudio_impl(logger.as_ref());
}

#[uniffi::export]
pub fn setAudioSettings(frequency: f32, incidence: i32, logger: Box<dyn LogCallbacks>) {
	logger.log_info_callback("Updating audio settings".to_string());

	let mut settings = SPATIAL_AUDIO_SETTINGS.write().unwrap();
	settings.frequency = frequency;
	settings.buffer_duration = 1.0 / (incidence as f32);
}

#[uniffi::export]
pub fn setDepthAudioPaused(paused: bool, logger: Box<dyn LogCallbacks>) {
	logger.log_info_callback(format!("Setting depth audio playback. Paused: {}", paused));

	SPATIAL_AUDIO_SETTINGS.write().unwrap().depth_audio_paused = paused;
}

#[uniffi::export]
pub fn setObjectAudioPaused(paused: bool, logger: Box<dyn LogCallbacks>) {
	logger.log_info_callback(format!("Setting object audio playback. Paused: {}", paused));

	SPATIAL_AUDIO_SETTINGS.write().unwrap().object_audio_paused = paused;
}

#[uniffi::export]
pub fn sendAIDataForSpatialAudio(
	mut depth_data_buffer: UniffiFloatBufferWrapper,
	object_data_buffer: Vec<UniffiDetectedObject>,
	logger: Box<dyn LogCallbacks>,
) {
	let depth_data_buffer = depth_data_buffer.as_slice_mut();
	let depth_estimation_data: &[f32; 256 * 256] = depth_data_buffer
		.as_ref()
		.try_into()
		.expect("depth_data_buffer needs to be 256x256!");

	let logger = Arc::new(logger);
	let logger_clone1 = logger.clone();
	let logger_clone2 = logger.clone();

	let object_detection_data = object_data_buffer
		.into_iter()
		.map(|o| o.into())
		.collect::<Vec<TrackedObject>>();

	let should_restart = try_change_spatial_audio(|spatial_audio| {
		spatial_audio.update(
			depth_estimation_data,
			&object_detection_data,
			Arc::new(move |msg| logger.log_info_callback(msg.to_string())),
			Arc::new(move |msg| logger_clone1.log_error_callback(msg.to_string())),
		)
	});
	if let Some(should_restart) = should_restart
		&& should_restart
	{
		createSpatialAudio_impl(logger_clone2.as_ref().as_ref());
	}
}

#[uniffi::export]
pub fn destroySpatialAudio(logger: Box<dyn LogCallbacks>) {
	logger.log_info_callback("shutting down spatial audio...".to_string());

	*SPATIAL_AUDIO.write().unwrap() = None;
}

uniffi::setup_scaffolding!("NativeLib");
