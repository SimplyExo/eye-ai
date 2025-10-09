use crate::audio::Vec3;
use alto::Mono;

pub struct DepthAudioSourceData {
	pub frequency: f32,
	pub duration: f32,
	pub sample_rate: usize,
	pub position: Vec3,
	pub samples: Vec<Mono<i16>>,
}

impl DepthAudioSourceData {
	pub fn new(frequency: f32, duration: f32, sample_rate: usize, position: Vec3) -> Self {
		Self {
			frequency,
			duration,
			sample_rate,
			position,
			samples: create_audio_data(frequency, duration, sample_rate),
		}
	}
}

fn create_audio_data(base_frequency: f32, duration: f32, sample_rate: usize) -> Vec<Mono<i16>> {
	const TWO_PI: f32 = std::f32::consts::PI * 2.0;

	let num_samples = number_of_samples(sample_rate, duration);
	let amplitude: f32 = 0.8;
	let decay_rate: f32 = 8.0;

	let mut samples: Vec<Mono<i16>> = Vec::new();
	samples.resize(num_samples, Mono { center: 0 });
	for (i, sample) in samples.iter_mut().enumerate() {
		let t = i as f32 / sample_rate as f32;

		let envelope = (-decay_rate * t).exp();

		let wave = (TWO_PI * base_frequency * t).sin()
			+ 0.5 * (TWO_PI * base_frequency * 2.0 * t).sin()
			+ 0.25 * (TWO_PI * base_frequency * 3.0 * t).sin();

		*sample = Mono {
			center: ((amplitude * envelope * wave * 32760.0) / 1.75) as i16,
		};
	}

	samples
}

fn number_of_samples(sample_rate: usize, duration: f32) -> usize {
	(sample_rate as f32 * duration) as usize
}
