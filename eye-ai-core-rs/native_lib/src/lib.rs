#![allow(non_snake_case)]

use eye_ai_core_rs::{
	BoundingBox, CreateDepthModelInfo, CreateYoloModelInfo, DepthModelNpuConfig, DetectedObject,
	FloatTensorBuffer, FloatTensorFormat, MetricDepthModel, ObjectTracker, ProfilingFrame,
	TrackedObject, YoloModel, YoloModelNpuConfig,
	audio::{SpatialAudio, SpatialAudioContent, read_audio_file, read_object_label_data},
	inferno_colormap,
};
use eye_ai_core_rs_profiling_attribute::profile_function;
use std::{
	ffi::CString,
	sync::{Arc, LazyLock, RwLock, atomic::Ordering},
};
use tracing::{debug, error, trace};

mod audio_session;
use audio_session::AudioSessions;

#[cfg(target_os = "android")]
mod android_logging;
#[cfg(target_os = "android")]
use android_logging::AndroidLogLayer;

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

static SPATIAL_AUDIO: LazyLock<AudioSessions<SpatialAudio>> = LazyLock::new(AudioSessions::default);
static SPATIAL_AUDIO_CONTENT: LazyLock<RwLock<Option<Arc<SpatialAudioContent>>>> =
	LazyLock::new(|| RwLock::new(None));
// Audio has no native finish/drain consumer, so completed scopes must not be
// retained for the lifetime of this process.
static AUDIO_PROFILING_FRAME: LazyLock<Arc<ProfilingFrame>> =
	LazyLock::new(|| Arc::new(ProfilingFrame::new_unretained("Audio")));

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

fn try_change_spatial_audio(session_id: u64, f: impl FnOnce(&mut SpatialAudio) -> bool) {
	let Some(session) = SPATIAL_AUDIO.get(session_id) else {
		return;
	};
	if !session.is_active() {
		return;
	}
	// Only clone resource data under its lock; opening a device must not block
	// settings/content updates or any other session.
	let content = SPATIAL_AUDIO_CONTENT.read().unwrap().clone();
	let result = session.change(
		|| {
			let content = content
				.clone()
				.ok_or("audio content is not configured".to_owned())?;
			SpatialAudio::new_in_session(
				session.settings.clone(),
				content,
				AUDIO_PROFILING_FRAME.clone(),
				session.active.clone(),
				session.object_audio_playback_epoch.clone(),
			)
			.map_err(|error| error.to_string())
		},
		f,
	);
	if let Err(error) = result {
		// A temporary missing device is retried by the next send in THIS session.
		error!(session_id, "Failed to create spatial audio: {error}");
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
impl std::fmt::Debug for UniffiFloatBufferWrapper {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("UniffiFloatBufferWrapper")
			.field("ptr", &(self.ptr_address as *mut f32))
			.field("len", &self.length)
			.finish()
	}
}
impl<const N: usize> From<&mut [f32; N]> for UniffiFloatBufferWrapper {
	fn from(value: &mut [f32; N]) -> Self {
		Self {
			ptr_address: value.as_mut_ptr() as i64,
			length: N as i32,
		}
	}
}

#[derive(Debug, uniffi::Record)]
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

// TODO: maybe put this into JNI_OnLoad?
/// MUST BE CALLED AT THE START OF YOUR ANDROID APP!
#[cfg(target_os = "android")]
#[uniffi::export]
pub fn initAndroidLogging() {
	use tracing_subscriber::Registry;
	use tracing_subscriber::layer::SubscriberExt;

	tracing::subscriber::set_global_default(Registry::default().with(AndroidLogLayer)).unwrap();

	#[cfg(feature = "enable_tracy_profiling")]
	tracing_tracy::client::Client::start();
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
pub fn initMetricDepthModel(
	model_name: String,
	relative_depth_model: Vec<u8>,
	enable_npu: bool,
	skel_directory: String,
) {
	debug!(
		model_name = ?model_name,
		enable_npu = ?enable_npu,
		skel_directory = ?skel_directory,
		"initMetricDepthModel()",
	);

	let metric_depth_model = MetricDepthModel::new(
		CreateDepthModelInfo {
			model_name,
			model_data: relative_depth_model,
			npu_config: if enable_npu {
				Some(DepthModelNpuConfig {
					skel_library_dir: CString::new(skel_directory).unwrap(),
				})
			} else {
				None
			},
		},
		&DEPTH_PROFILING_FRAME,
	)
	.expect("failed to create metric depth model");

	debug!("finished creating metric depth model");

	*METRIC_DEPTH_MODEL.write().unwrap() = Some(metric_depth_model);
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
pub fn shutdownMetricDepthModel() {
	debug!("shutdownMetricDepthModel()");

	*METRIC_DEPTH_MODEL.write().unwrap() = None;
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
pub fn runMetricDepthModelInference(
	mut input: UniffiFloatBufferWrapper,
	mut output: UniffiFloatBufferWrapper,
) {
	wait_for_metric_depth_model(|metric_depth_model| {
		match metric_depth_model.run(
			&mut FloatTensorBuffer::new(input.as_slice_mut(), FloatTensorFormat::MiDaSImageRgb),
			&mut FloatTensorBuffer::new(output.as_slice_mut(), FloatTensorFormat::MetricDepth),
		) {
			Ok(()) => {}
			Err(e) => {
				error!(
					"Failed to run metric depth model: {}, output will not be changed",
					e
				);
			}
		}
	});
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
pub fn getMetricDepthModelInputShape() -> Vec<i32> {
	wait_for_metric_depth_model(
		|metric_depth_model| match metric_depth_model.get_input_shape() {
			Some(input_shape) => input_shape,
			None => {
				error!("could not get input shape (returning empty vector)");
				vec![]
			}
		},
	)
}

#[uniffi::export]
#[profile_function("DEPTH_PROFILING_FRAME")]
pub fn getMetricDepthModelOutputShape() -> Vec<i32> {
	wait_for_metric_depth_model(
		|metric_depth_model| match metric_depth_model.get_output_shape() {
			Some(output_shape) => output_shape,
			None => {
				error!("could not get output shape (returning empty vector)");
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
) {
	const MAX_METRIC_DISTANCE: f32 = 5.0;

	let depth_buffer = depth_buffer.as_slice_mut();
	let colormapped_pixels = colormapped_pixels.as_slice_mut();
	if depth_buffer.len() == colormapped_pixels.len() {
		for (i, depth_value) in depth_buffer.iter().enumerate() {
			colormapped_pixels[i] = inferno_colormap(depth_value / MAX_METRIC_DISTANCE);
		}
	} else {
		error!(
			"depth_buffer and colormapped_pixels have different sizes: {} and {} (not changing colormapped_pixels!)",
			depth_buffer.len(),
			colormapped_pixels.len()
		);
	}
}

#[uniffi::export]
#[profile_function("OBJECT_PROFILING_FRAME")]
pub fn initYoloRuntime(
	model_name: String,
	model: Vec<u8>,
	labels: Vec<String>,
	enable_npu: bool,
	skel_directory: String,
) {
	debug!(
		model_name = ?model_name,
		enable_npu = ?enable_npu,
		skel_directory = ?skel_directory,
		"initYoloRuntime"
	);

	let yolo_model = YoloModel::new(
		CreateYoloModelInfo {
			model_name,
			labels: labels.clone(),
			model_data: model,
			npu_config: if enable_npu {
				Some(YoloModelNpuConfig {
					skel_library_dir: CString::new(skel_directory).unwrap(),
				})
			} else {
				None
			},
		},
		&OBJECT_PROFILING_FRAME,
	)
	.expect("failed to create yolo model");

	debug!("created yolo model");

	// Publish model + tracker atomically with respect to inference and reset.
	// Native lock order is always YOLO_MODEL -> OBJECT_TRACKER.
	let mut model_slot = YOLO_MODEL.write().unwrap();
	let object_tracker = ObjectTracker::new(labels, &OBJECT_PROFILING_FRAME);
	*OBJECT_TRACKER.write().unwrap() = Some(object_tracker);
	*model_slot = Some(yolo_model);
}

/// Reset only tracking evidence, never the detector. Taking the model write lock
/// waits for the ENTIRE old runYoloOperation, including its final tracker update.
/// Callers serialize reset + the next inference together (Kotlin's model session).
#[uniffi::export]
pub fn resetObjectTracker() {
	let _model = YOLO_MODEL.write().unwrap();
	if let Some(tracker) = OBJECT_TRACKER.write().unwrap().as_mut() {
		tracker.reset();
	}
}

#[derive(uniffi::Record, Clone, Debug)]
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
pub fn runYoloOperation(mut input: UniffiFloatBufferWrapper) -> Vec<UniffiDetectedObject> {
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
				error!(
					"Failed to run yolo model: {}, returning empty float array",
					e
				);
				vec![]
			}
		}
	})
}

#[uniffi::export]
#[profile_function("OBJECT_PROFILING_FRAME")]
pub fn getYoloInputShape() -> Vec<i32> {
	wait_for_yolo_model(|yolo_model| match yolo_model.get_input_shape() {
		Some(input_shape) => input_shape,
		None => {
			error!("could not get input shape (returning empty vector)");
			vec![]
		}
	})
}

#[uniffi::export]
#[profile_function("OBJECT_PROFILING_FRAME")]
pub fn getYoloOutputShape() -> Vec<i32> {
	wait_for_yolo_model(|yolo_model| match yolo_model.get_output_shape() {
		Some(output_shape) => output_shape,
		None => {
			error!("could not get output shape (returning empty vector)");
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
#[profile_function("AUDIO_PROFILING_FRAME")]
pub fn setupAudioContent(
	coco_labels_audio_file_content: Vec<u8>,
	coco_labels_json_content: String,
) {
	debug!("setupAudioContent()");

	let coco_audio_file = match read_audio_file(
		coco_labels_audio_file_content.into_boxed_slice(),
		&AUDIO_PROFILING_FRAME,
	) {
		Ok(coco_audio_file) => coco_audio_file,
		Err(e) => {
			error!("Failed to read coco labels audio file: {e}");
			return;
		}
	};

	let coco_labels_data =
		match read_object_label_data(&coco_labels_json_content, &AUDIO_PROFILING_FRAME) {
			Ok(coco_labels_data) => coco_labels_data,
			Err(e) => {
				error!("Failed to read object label json data: {e}");
				return;
			}
		};

	let content = Arc::new(SpatialAudioContent::new(coco_audio_file, coco_labels_data));

	SPATIAL_AUDIO_CONTENT
		.write()
		.unwrap()
		.replace(content.clone());
}

/// Allocate identity only: no device work and no engine lock on the calling thread.
#[uniffi::export]
pub fn beginSpatialAudioSession() -> u64 {
	SPATIAL_AUDIO.begin()
}

/// Stop boundary, safe on the main thread even during blocked create/send/destroy.
#[uniffi::export]
pub fn invalidateSpatialAudioSession(session_id: u64) {
	SPATIAL_AUDIO.invalidate(session_id);
}

/// Worker only. Content must have been configured by setupAudioContent.
#[uniffi::export]
#[profile_function("AUDIO_PROFILING_FRAME")]
pub fn createSpatialAudio(session_id: u64) {
	try_change_spatial_audio(session_id, |_| false);
}

#[uniffi::export]
#[profile_function("AUDIO_PROFILING_FRAME")]
pub fn setAudioSettings(session_id: u64, frequency: f32, incidence: i32) {
	trace!(
		frequency = ?frequency,
		incidence = ?incidence,
		"setAudioSettings()"
	);

	let Some(session) = SPATIAL_AUDIO.get(session_id) else {
		return;
	};
	let mut settings = session.settings.write().unwrap();
	if !session.is_active() {
		return;
	}
	settings.frequency = frequency;
	settings.buffer_duration = 1.0 / (incidence as f32);
}

#[uniffi::export]
#[profile_function("AUDIO_PROFILING_FRAME")]
pub fn setDepthAudioPaused(session_id: u64, paused: bool) {
	trace!(paused = ?paused, "setDepthAudioPaused()");

	let Some(session) = SPATIAL_AUDIO.get(session_id) else {
		return;
	};
	let mut settings = session.settings.write().unwrap();
	if session.is_active() {
		settings.depth_audio_paused = paused;
	}
}

#[uniffi::export]
#[profile_function("AUDIO_PROFILING_FRAME")]
pub fn setObjectAudioPaused(session_id: u64, paused: bool) {
	trace!(paused = ?paused, "setObjectAudioPaused()");

	let Some(session) = SPATIAL_AUDIO.get(session_id) else {
		return;
	};
	let mut settings = session.settings.write().unwrap();
	if session.is_active() {
		settings.object_audio_paused = paused;
		if paused {
			// Pausing is an explicit interruption, unlike a transient empty snapshot.
			session
				.object_audio_playback_epoch
				.fetch_add(1, Ordering::AcqRel);
		}
	}
}

/// Hard boundary for a stream/content generation change. Unlike an empty object
/// snapshot this interrupts a currently playing object announcement. The token is
/// session-local, so a delayed call cannot affect a later audio session.
#[uniffi::export]
#[profile_function("AUDIO_PROFILING_FRAME")]
pub fn invalidateObjectAudioPlayback(session_id: u64) {
	let Some(session) = SPATIAL_AUDIO.get(session_id) else {
		return;
	};
	if session.is_active() {
		session
			.object_audio_playback_epoch
			.fetch_add(1, Ordering::AcqRel);
	}
}

#[uniffi::export]
#[profile_function("AUDIO_PROFILING_FRAME")]
pub fn sendAIDataForSpatialAudio(
	session_id: u64,
	mut depth_data_buffer: UniffiFloatBufferWrapper,
	object_data_buffer: Vec<UniffiDetectedObject>,
) {
	if !SPATIAL_AUDIO
		.get(session_id)
		.is_some_and(|session| session.is_active())
	{
		return;
	}
	let depth_data_buffer = depth_data_buffer.as_slice_mut();
	let depth_estimation_data: &[f32; 256 * 256] = depth_data_buffer
		.as_ref()
		.try_into()
		.expect("depth_data_buffer needs to be 256x256!");

	let object_detection_data = object_data_buffer
		.into_iter()
		.map(|o| o.into())
		.collect::<Vec<TrackedObject>>();

	try_change_spatial_audio(session_id, |spatial_audio| {
		spatial_audio.update(depth_estimation_data, &object_detection_data)
	});
}

#[uniffi::export]
#[profile_function("AUDIO_PROFILING_FRAME")]
pub fn destroySpatialAudio(session_id: u64) {
	debug!(session_id, "destroySpatialAudio()");
	SPATIAL_AUDIO.destroy(session_id);
}

uniffi::setup_scaffolding!("NativeLib");

#[cfg(test)]
mod audio_export_tests {
	use super::*;
	use std::{sync::mpsc, thread, time::Duration};

	#[test]
	fn delayed_settings_and_stale_exports_are_bound_to_their_original_session() {
		let a = beginSpatialAudioSession();
		let old = SPATIAL_AUDIO.get(a).unwrap();
		let settings_guard = old.settings.write().unwrap();
		let (entered_tx, entered_rx) = mpsc::channel();
		let settings_call = thread::spawn(move || {
			entered_tx.send(()).unwrap();
			setAudioSettings(a, 999.0, 3);
			setDepthAudioPaused(a, false);
			setObjectAudioPaused(a, false);
		});
		entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
		invalidateSpatialAudioSession(a);
		let b = beginSpatialAudioSession();
		let b_playback_epoch = SPATIAL_AUDIO
			.get(b)
			.unwrap()
			.object_audio_playback_epoch
			.clone();
		assert_eq!(b_playback_epoch.load(Ordering::Acquire), 0);
		setAudioSettings(b, 321.0, 5);
		setDepthAudioPaused(b, true);
		setObjectAudioPaused(b, true);
		assert_eq!(b_playback_epoch.load(Ordering::Acquire), 1);
		invalidateObjectAudioPlayback(b);
		assert_eq!(b_playback_epoch.load(Ordering::Acquire), 2);
		invalidateObjectAudioPlayback(a);
		assert_eq!(b_playback_epoch.load(Ordering::Acquire), 2);
		drop(settings_guard);
		settings_call.join().unwrap();
		createSpatialAudio(a); // Must not try opening a device, even without content.
		let mut depth = [1_000.0; 256 * 256];
		sendAIDataForSpatialAudio(a, (&mut depth).into(), vec![]);
		destroySpatialAudio(a);
		destroySpatialAudio(a);
		let fresh = SPATIAL_AUDIO.get(b).unwrap();
		assert!(fresh.is_active());
		let settings = fresh.settings.read().unwrap();
		assert_eq!(settings.frequency, 321.0);
		assert!(settings.depth_audio_paused && settings.object_audio_paused);
		drop(settings);
		destroySpatialAudio(b);
		assert!(SPATIAL_AUDIO.get(a).is_none());
		assert!(SPATIAL_AUDIO.get(b).is_none());
	}
}

#[cfg(test)]
mod tracking_epoch_tests {
	use super::*;
	use std::{sync::mpsc, thread, time::Duration};

	fn detection() -> DetectedObject {
		DetectedObject::new(
			"person".to_owned(),
			0,
			1.0,
			BoundingBox::new(0.5, 0.5, 0.2, 0.2),
		)
	}

	#[test]
	fn reset_seam_waits_for_old_native_mutation_and_clears_confirmed_evidence() {
		// No detector is needed: exercise the production locks and reset export,
		// with real ByteTrack/validation and a barrier in place of detector work.
		assert!(YOLO_MODEL.read().unwrap().is_none());
		*OBJECT_TRACKER.write().unwrap() = Some(ObjectTracker::new(
			vec!["person".to_owned()],
			&OBJECT_PROFILING_FRAME,
		));
		let mut confirmed = vec![];
		for _ in 0..5 {
			confirmed = OBJECT_TRACKER
				.write()
				.unwrap()
				.as_mut()
				.unwrap()
				.update(vec![detection()]);
			thread::sleep(Duration::from_millis(150));
		}
		assert_eq!(confirmed.len(), 1);
		assert_eq!(confirmed[0].tracking_id, 1);

		let (entered_tx, entered_rx) = mpsc::channel();
		let (release_tx, release_rx) = mpsc::channel();
		let old = thread::spawn(move || {
			// Same lifetime as wait_for_yolo_model in runYoloOperation:
			// the model guard survives until AFTER tracker mutation.
			let _model = YOLO_MODEL.write().unwrap();
			entered_tx.send(()).unwrap();
			release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
			let result = OBJECT_TRACKER
				.write()
				.unwrap()
				.as_mut()
				.unwrap()
				.update(vec![detection()]);
			assert_eq!(result.len(), 1, "old completion still belongs to A");
		});
		entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
		let (requested_tx, requested_rx) = mpsc::channel();
		let (reset_tx, reset_rx) = mpsc::channel();
		let reset = thread::spawn(move || {
			requested_tx.send(()).unwrap();
			resetObjectTracker();
			reset_tx.send(()).unwrap();
		});
		requested_rx.recv_timeout(Duration::from_secs(3)).unwrap();
		assert!(matches!(
			reset_rx.recv_timeout(Duration::from_millis(100)),
			Err(mpsc::RecvTimeoutError::Timeout)
		));
		release_tx.send(()).unwrap();
		old.join().unwrap();
		reset_rx.recv_timeout(Duration::from_secs(3)).unwrap();
		reset.join().unwrap();

		let _model = YOLO_MODEL.write().unwrap();
		let mut tracker_slot = OBJECT_TRACKER.write().unwrap();
		let first_b = tracker_slot.as_mut().unwrap().update(vec![detection()]);
		assert!(
			first_b.is_empty(),
			"B at the same position must be TENTATIVE"
		);
		assert!(_model.is_none(), "reset must never load a detector");
		*tracker_slot = None;
	}
}
