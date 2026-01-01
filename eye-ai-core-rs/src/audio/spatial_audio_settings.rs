use crate::audio::IVec2;

#[derive(Debug, Clone)]
pub struct SpatialAudioSettings {
	pub depth_audio_paused: bool,
	pub object_audio_paused: bool,
	pub frequency: f32,
	pub buffer_duration: f32,
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

	pub fn new(frequency: f32, buffer_duration: f32) -> Self {
		Self {
			depth_audio_paused: false,
			object_audio_paused: false,
			frequency,
			buffer_duration,
		}
	}
}
impl Default for SpatialAudioSettings {
	fn default() -> Self {
		Self::new(Self::DEFAULT_FREQUENCY, Self::DEFAULT_BUFFER_DURATION)
	}
}
