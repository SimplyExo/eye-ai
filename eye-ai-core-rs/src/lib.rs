//! # eye-ai-core-rs
//!
//! The core multi-platform library that implements most of EyeAI's features.
//!
//! Used by the Android App ("EyeAIApp"), as well as desktop tests,
//! which makes supporting new platforms super simple.
//!
//! See [Additional Documentation](additional_documentation) for more information.

#[cfg(any(doc, doctest))]
#[doc = include_str!("../doc/Additional Documentation.md")]
pub mod additional_documentation {
	#[doc = include_str!("../../README.md")]
	pub mod eye_ai_readme {}

	#[doc = include_str!("../README.md")]
	pub mod eye_ai_core_rs_readme {}

	#[doc = include_str!("../doc/GoogleAIStudioReadMe.md")]
	pub mod google_ai_studio_readme {}

	#[doc = include_str!("../doc/OCRReadMe.md")]
	pub mod ocr_readme {}

	#[doc = include_str!("../doc/SpatialAudioReadMe.md")]
	pub mod spatial_audio_readme {}

	#[doc = include_str!("../doc/SpeechRecognitionReadMe.md")]
	pub mod speech_recognition_readme {}

	#[doc = include_str!("../doc/StateMachineReadMe.md")]
	pub mod state_machine_readme {}

	#[doc = include_str!("../doc/TTSEngineReadMe.md")]
	pub mod tts_engine_readme {}
}

pub mod audio;

pub mod litert;

mod tensor_buffer;
pub use tensor_buffer::{
	FloatTensorBuffer, FloatTensorFormat, TensorBuffer, TensorBufferContainer,
	check_float_tensor_format, image_rgb_255_to_midas_image,
};

mod depth_model;
pub use depth_model::{CreateDepthModelInfo, DepthModel, DepthModelNpuConfig};

mod metric_depth_model;
pub use metric_depth_model::MetricDepthModel;

mod yolo_model;
pub use yolo_model::{
	BoundingBox, CreateYoloModelInfo, DetectedObject, YoloModel, YoloModelNpuConfig,
};

mod object_tracker;
pub use object_tracker::{ObjectTracker, TrackedObject};

mod profiling;
pub use profiling::ProfilingFrame;

mod colormap;
pub use colormap::inferno_colormap;
