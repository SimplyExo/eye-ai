use crate::audio::Vec3;

#[derive(Debug, Clone)]
pub struct ObjectAudioSourceData {
	pub object_id: usize,
	pub name: String,
	pub sound_begin: usize,
	pub sound_end: usize,
	pub position: Vec3,
}
