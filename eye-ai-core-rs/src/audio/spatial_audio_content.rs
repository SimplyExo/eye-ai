use crate::ProfilingFrame;
use eye_ai_core_rs_profiling_attribute::profile_function;
use json::JsonValue;
use std::collections::HashMap;
use std::io::Cursor;
use symphonia::core::{
	audio::SampleBuffer, codecs::DecoderOptions, errors::Error, formats::FormatOptions,
	io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};
use tracing::debug;

#[derive(Debug)]
pub struct SpatialAudioContent {
	pub coco_labels_audio_file: AudioFileData,
	pub object_label_data: HashMap<String, ObjectLabelData>,
}
impl SpatialAudioContent {
	pub fn new(
		coco_labels_audio_file: AudioFileData,
		object_label_data: HashMap<String, ObjectLabelData>,
	) -> Self {
		Self {
			coco_labels_audio_file,
			object_label_data,
		}
	}
}

#[derive(Debug)]
pub struct ObjectLabelData {
	pub sample_begin: usize,
	pub sample_end: usize,
}
#[profile_function("profiling_frame")]
pub fn read_object_label_data(
	json_content: &str,
	profiling_frame: &ProfilingFrame,
) -> Result<HashMap<String, ObjectLabelData>, json::Error> {
	debug!("read_object_label_data()");

	let json = json::parse(json_content)?;
	let JsonValue::Array(json_array) = json else {
		return Err(json::Error::WrongType("json should be array!".to_string()));
	};
	let mut map = HashMap::<String, ObjectLabelData>::new();
	for json_data in json_array {
		let JsonValue::Object(data_object) = json_data else {
			return Err(json::Error::wrong_type("array element should be object!"));
		};
		let object_label_data = ObjectLabelData {
			sample_begin: data_object["start"]
				.as_usize()
				.ok_or(json::Error::wrong_type("usize"))?,
			sample_end: data_object["end"]
				.as_usize()
				.ok_or(json::Error::wrong_type("usize"))?,
		};
		let label = data_object["text"]
			.as_str()
			.ok_or(json::Error::wrong_type("string"))?;
		map.insert(label.to_string().to_lowercase(), object_label_data);
	}
	Ok(map)
}

pub struct AudioFileData {
	pub samples: Vec<i16>,
	pub sample_rate: u32,
}
impl std::fmt::Debug for AudioFileData {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("AudioFileData")
			.field("sample_rate", &self.sample_rate)
			.field(
				"samples",
				&format!("(hidden Vec<i16>) len={}", self.samples.len()),
			)
			.finish_non_exhaustive()
	}
}

#[profile_function("profiling_frame")]
pub fn read_audio_file(
	audio_file_content: Box<[u8]>,
	profiling_frame: &ProfilingFrame,
) -> Result<AudioFileData, Error> {
	debug!("read_audio_file()");

	let mss = MediaSourceStream::new(
		Box::new(Cursor::new(audio_file_content)),
		Default::default(),
	);

	let hint = Hint::new();
	let format_opts = FormatOptions::default();
	let metadata_opts = MetadataOptions::default();
	let decoder_opts = DecoderOptions::default();

	let probed =
		symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
	let mut format = probed.format;
	let track = format.default_track().unwrap();
	let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;
	let sample_rate = track.codec_params.sample_rate.unwrap();

	let mut samples = Vec::new();

	while let Ok(packet) = format.next_packet() {
		if let Ok(audio_buf) = decoder.decode(&packet) {
			let mut sample_buf =
				SampleBuffer::<i16>::new(audio_buf.capacity() as u64, *audio_buf.spec());

			sample_buf.copy_interleaved_ref(audio_buf);

			samples.extend(sample_buf.samples());
		}
	}

	Ok(AudioFileData {
		samples,
		sample_rate,
	})
}
