use crate::audio::IVec2;

pub type LogCallback = fn(&str);

#[derive(Debug, Clone)]
pub struct SpatialAudioSettings {
	pub coco_labels_audio_file_content: Vec<u8>,
	pub coco_labels_json_content: String,

	pub depth_audio_paused: bool,
	pub object_audio_paused: bool,
	pub frequency: f32,
	pub buffer_duration: f32,

	pub log_info_callback: LogCallback,
	pub log_error_callback: LogCallback,
}

impl SpatialAudioSettings {
	pub const DEFAULT_FREQUENCY: f32 = 500.0;
	pub const DEFAULT_BUFFER_DURATION: f32 = 0.25;

	pub const BUFFERS_PER_SOURCE: usize = 3;
	pub const SAMPLE_RATE: usize = 48000;
	pub const PICTURE_RESOLUTION: IVec2 = IVec2 { x: 256, y: 256 };
	pub const NUMBER_OF_SOURCES: usize = 9;
	pub const MAX_DISTANCE: f32 = 2.5;
	pub const ROLLOFF_FACTOR: f32 = 1.0;
	pub const REFERENCE_DISTANCE: f32 = 1.0;

	pub fn new(
		coco_labels_audio_file_content: Vec<u8>,
		coco_labels_json_content: String,
		log_info_callback: LogCallback,
		log_error_callback: LogCallback,
	) -> Self {
		Self {
			coco_labels_json_content,
			coco_labels_audio_file_content,
			depth_audio_paused: false,
			object_audio_paused: false,
			frequency: Self::DEFAULT_FREQUENCY,
			buffer_duration: Self::DEFAULT_BUFFER_DURATION,
			log_info_callback,
			log_error_callback,
		}
	}
}
