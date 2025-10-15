use eye_ai_core_rs_profiling_attribute::profile_function;
use std::io::Cursor;
use symphonia::core::{
	audio::SampleBuffer, codecs::DecoderOptions, errors::Error, formats::FormatOptions,
	io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

pub struct AudioFileData {
	pub samples: Vec<i16>,
	pub sample_rate: u32,
}

#[profile_function]
pub fn read_audio_file(audio_file_content: Box<[u8]>) -> Result<AudioFileData, Error> {
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
