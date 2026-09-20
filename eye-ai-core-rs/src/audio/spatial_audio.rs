use alto::{
	Alto, Context, ContextAttrs, DeviceObject, DistanceModel, Mono, OutputDevice, Source,
	SourceState, ext::Alc,
};
use eye_ai_core_rs_profiling_attribute::profile_function;
use std::{
	collections::{HashMap, VecDeque},
	sync::{
		Arc, RwLock,
		atomic::{AtomicBool, Ordering},
	},
	time::Duration,
};
use thiserror::Error;
use tracing::{debug, error, trace};
use tracing_tracy::client::set_thread_name;

use crate::{
	FormattedProfilingFrame, TrackedObject,
	audio::{
		CalculateSoundOrigin, DepthAudioSourceData, IVec2, ObjectAudioSourceData, ObjectLabelData,
		SpatialAudioContent, SpatialAudioSettings, Vec3, spatial_audio_content::AudioFileData,
	},
	profile_scope,
};

#[derive(Debug, Error)]
pub enum SpatialAudioError {
	#[error("Alto error: {0}")]
	AltoError(#[from] alto::AltoError),
	#[error("JSON error: {0}")]
	JsonError(#[from] json::Error),
}

pub struct SpatialAudio {
	_alto: Alto,
	_device: OutputDevice,
	_context: Arc<Context>,
	depth_audio_sources_data: Arc<RwLock<Vec<DepthAudioSourceData>>>,
	_depth_audio_thread: std::thread::JoinHandle<()>,
	depth_audio_running: Arc<AtomicBool>,
	object_audio_sources_data: Arc<RwLock<VecDeque<ObjectAudioSourceData>>>,
	_object_audio_thread: std::thread::JoinHandle<()>,
	object_audio_running: Arc<AtomicBool>,
	pub settings: Arc<RwLock<SpatialAudioSettings>>,
	content: Arc<SpatialAudioContent>,
	profiling_frame: Arc<FormattedProfilingFrame>,
}
impl std::fmt::Debug for SpatialAudio {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("SpatialAudio")
			.field("settings", &self.settings)
			.finish_non_exhaustive()
	}
}
impl Drop for SpatialAudio {
	fn drop(&mut self) {
		self.depth_audio_running.store(false, Ordering::Relaxed);
		self.object_audio_running.store(false, Ordering::Relaxed);
	}
}
impl SpatialAudio {
	#[profile_function("profiling_frame")]
	pub fn new(
		settings: Arc<RwLock<SpatialAudioSettings>>,
		content: Arc<SpatialAudioContent>,
		profiling_frame: Arc<FormattedProfilingFrame>,
		depth_audio_thread_profiling_frame: Arc<FormattedProfilingFrame>,
		object_audio_thread_profiling_frame: Arc<FormattedProfilingFrame>,
	) -> Result<Self, SpatialAudioError> {
		debug!(
			settings = ?settings,
			"new()"
		);

		let profiling_frame_clone = profiling_frame.clone();

		let settings_clone1 = settings.clone();
		let settings_clone2 = settings.clone();

		let content_clone = content.clone();

		let depth_audio_running = Arc::new(AtomicBool::new(true));
		let depth_audio_running_clone = depth_audio_running.clone();

		let object_audio_running = Arc::new(AtomicBool::new(true));
		let object_audio_running_clone = depth_audio_running.clone();

		let alto = Alto::load_default()?;
		let device = alto.open(None)?;
		if device.is_extension_present(Alc::SoftHrtf) {
			let attribs = ContextAttrs {
				soft_hrtf: Some(true),
				..Default::default()
			};
			device.soft_reset(Some(attribs)).unwrap();
		}
		let context = Arc::new(device.new_context(None)?);
		context.set_distance_model(DistanceModel::LinearClamped);
		context.set_position([0.0, 0.0, 0.0])?;
		let context_clone1 = context.clone();
		let context_clone2 = context.clone();

		let depth_audio_sources_data = Arc::new(RwLock::new(Vec::<DepthAudioSourceData>::new()));
		let depth_audio_sources_data_clone = depth_audio_sources_data.clone();

		let object_audio_sources_data =
			Arc::new(RwLock::new(VecDeque::<ObjectAudioSourceData>::new()));
		let object_audio_sources_data_clone = object_audio_sources_data.clone();

		Ok(Self {
			_alto: alto,
			_device: device,
			_context: context,
			depth_audio_sources_data,
			_depth_audio_thread: std::thread::Builder::new()
				.name("Depth Audio".to_string())
				.spawn(move || {
					depth_audio_thread(
						depth_audio_running_clone,
						context_clone1,
						depth_audio_sources_data_clone,
						settings_clone1,
						depth_audio_thread_profiling_frame,
					)
				})
				.expect("failed to spawn depth audio thread"),
			depth_audio_running,
			object_audio_sources_data,
			_object_audio_thread: std::thread::Builder::new()
				.name("Object Audio".to_string())
				.spawn(move || {
					object_audio_thread(
						object_audio_running_clone,
						settings_clone2,
						&content_clone.coco_labels_audio_file,
						context_clone2,
						object_audio_sources_data_clone,
						object_audio_thread_profiling_frame,
					)
				})
				.expect("failed to spawn object audio thread"),
			object_audio_running,
			settings,
			content,
			profiling_frame: profiling_frame_clone,
		})
	}

	/// returns whether it needs to be recreated, as the output device changed
	pub fn update(
		&mut self,
		depth_estimation_data: &[f32; SpatialAudioSettings::PICTURE_PIXEL_COUNT],
		object_detection_data: &[TrackedObject],
		segmentation_data: Option<&[i32; SpatialAudioSettings::PICTURE_PIXEL_COUNT]>,
		segmentation_class_importances: &[f32],
	) -> bool {
		let mut should_restart = false;

		profile_scope!(self.profiling_frame, "update");

		// this extension is implemented on android, such that we can restart SpatialAudio when the output device changes
		// on desktop (linux at least), this is not needed and the extension is not implemented, as output switching happens automatically
		if self._device.is_extension_present(Alc::Disconnect) {
			match self._device.connected() {
				Ok(connected) => {
					if !connected {
						error!("No audio device connected right now!");
						should_restart = true;
					}
				}
				Err(e) => {
					error!(
						"Failed to retrieve if device is connected, even though ALC_EXT_disconnect is present: {e}"
					);
				}
			}
		}

		let settings = self.settings.read().unwrap();
		let depth_audio_paused = settings.depth_audio_paused;
		let object_audio_paused = settings.object_audio_paused;

		if !depth_audio_paused {
			*self.depth_audio_sources_data.write().unwrap() = process_depth_estimation_data(
				depth_estimation_data,
				segmentation_data,
				segmentation_class_importances,
				&settings,
				&self.profiling_frame,
			);
		}
		if !object_audio_paused {
			let new_audio_sources_data = process_object_detection_data(
				depth_estimation_data,
				object_detection_data,
				&self.content.object_label_data,
				&self.profiling_frame,
			);
			let mut object_audio_sources_data = self.object_audio_sources_data.write().unwrap();
			for new_source_data in new_audio_sources_data {
				let mut found = false;
				for source_data in object_audio_sources_data.iter_mut() {
					if source_data.object_id == new_source_data.object_id {
						found = true;
						*source_data = new_source_data.clone();
						break;
					}
				}
				if !found {
					// max of 6 sources at once
					while object_audio_sources_data.len() >= 6 {
						object_audio_sources_data.pop_front();
					}
					object_audio_sources_data.push_back(new_source_data);
				}
			}
		}

		should_restart
	}
}

fn depth_audio_thread(
	running: Arc<AtomicBool>,
	context: Arc<Context>,
	depth_audio_sources_data: Arc<RwLock<Vec<DepthAudioSourceData>>>,
	settings: Arc<RwLock<SpatialAudioSettings>>,
	profiling_frame: Arc<FormattedProfilingFrame>,
) {
	set_thread_name!("Depth Audio");

	debug!(
		sample_rate = SpatialAudioSettings::SAMPLE_RATE,
		buffer_duration = settings.read().unwrap().buffer_duration,
		"depth_audio_thread()"
	);

	let mut sources = (0..SpatialAudioSettings::NUMBER_OF_SOURCES)
		.map(|_| context.new_streaming_source().unwrap())
		.collect::<Vec<_>>();

	for source in sources.iter_mut() {
		source
			.set_max_distance(SpatialAudioSettings::MAX_DISTANCE)
			.unwrap();
		source
			.set_rolloff_factor(SpatialAudioSettings::ROLLOFF_FACTOR)
			.unwrap();
		source
			.set_reference_distance(SpatialAudioSettings::REFERENCE_DISTANCE)
			.unwrap();
		source.set_gain(0.5).unwrap();
	}

	let silent_audio_source_data = DepthAudioSourceData::new(
		0.0,
		settings.read().unwrap().buffer_duration,
		SpatialAudioSettings::SAMPLE_RATE,
		Vec3::default(),
		0.0,
		&profiling_frame,
	);

	{
		let depth_audio_sources_data = depth_audio_sources_data.read().unwrap();

		for (i, source) in sources.iter_mut().enumerate() {
			let source_data = depth_audio_sources_data
				.get(i)
				.unwrap_or(&silent_audio_source_data);

			// fill all buffers with the same samples for now
			for _ in 0..SpatialAudioSettings::BUFFERS_PER_SOURCE {
				let buffer = context
					.new_buffer(&source_data.samples, source_data.sample_rate as i32)
					.unwrap();
				source.queue_buffer(buffer).unwrap();
			}
			source.set_position(source_data.position).unwrap();
			source.play();
		}
	}

	while running.load(Ordering::Relaxed) {
		{
			profile_scope!(profiling_frame, "Wait for resume");

			while settings.read().unwrap().depth_audio_paused {
				std::thread::sleep(Duration::from_millis(500));
			}
		}

		{
			profile_scope!(profiling_frame, "Filling buffers");

			let depth_audio_sources_data = depth_audio_sources_data.read().unwrap();

			for (i, source) in sources.iter_mut().enumerate() {
				let source_data = depth_audio_sources_data
					.get(i)
					.unwrap_or(&silent_audio_source_data);

				if source.buffers_processed() > 0 {
					let mut unqueued_buffer = source.unqueue_buffer().unwrap();
					unqueued_buffer
						.set_data(&source_data.samples, source_data.sample_rate as i32)
						.unwrap();
					source.queue_buffer(unqueued_buffer).unwrap();
				}
				source.set_position(source_data.position).unwrap();
				source.set_gain(source_data.gain).unwrap();
				source
					.set_max_distance(SpatialAudioSettings::MAX_DISTANCE)
					.unwrap();
				source
					.set_rolloff_factor(SpatialAudioSettings::ROLLOFF_FACTOR)
					.unwrap();
				source
					.set_reference_distance(SpatialAudioSettings::REFERENCE_DISTANCE)
					.unwrap();

				if source.state() == SourceState::Stopped {
					source.play();
				}
			}
		}

		{
			profile_scope!(profiling_frame, "Sleep 2ms");

			std::thread::sleep(Duration::from_millis(2));
		}

		profiling_frame.finish();
	}
}

fn object_audio_thread(
	running: Arc<AtomicBool>,
	settings: Arc<RwLock<SpatialAudioSettings>>,
	coco_audio_file: &AudioFileData,
	context: Arc<Context>,
	object_audio_sources_data: Arc<RwLock<VecDeque<ObjectAudioSourceData>>>,
	profiling_frame: Arc<FormattedProfilingFrame>,
) {
	set_thread_name!("Object Audio");

	let coco_audio_samples: &[i16] = &coco_audio_file.samples;

	debug!(
		sample_rate = coco_audio_file.sample_rate,
		"object_audio_thread()"
	);

	let mut source = context.new_static_source().unwrap();
	source.set_gain(1.0).unwrap();

	let mut sound_buffer: Vec<Mono<i16>> = Vec::new();

	while running.load(Ordering::Relaxed) {
		{
			profile_scope!(profiling_frame, "Wait for resume");

			while settings.read().unwrap().object_audio_paused {
				if !running.load(Ordering::Relaxed) {
					return;
				}
				std::thread::sleep(Duration::from_millis(500));
			}
		}

		let source_data = {
			profile_scope!(profiling_frame, "Wait for objects");

			loop {
				if !running.load(Ordering::Relaxed) {
					return;
				}
				if let Some(source_data) = object_audio_sources_data.write().unwrap().pop_front() {
					break source_data;
				}
				std::thread::sleep(Duration::from_millis(250));
			}
		};

		{
			profile_scope!(profiling_frame, "Fill audio buffers");

			let sample_rate_ms = coco_audio_file.sample_rate as usize / 1000;
			let duration_ms = source_data.sound_end - source_data.sound_begin;
			sound_buffer.resize(sample_rate_ms * duration_ms, Mono::<i16> { center: 0 });
			let begin_sample = sample_rate_ms * source_data.sound_begin;
			let end_sample = sample_rate_ms * source_data.sound_end;
			sound_buffer.copy_from_slice(
				coco_audio_samples[begin_sample..end_sample]
					.iter()
					.map(|sample| Mono::<i16> { center: *sample })
					.collect::<Vec<Mono<i16>>>()
					.as_slice(),
			);
			let buffer = Arc::new(
				context
					.new_buffer(&sound_buffer, coco_audio_file.sample_rate as i32)
					.unwrap(),
			);
			source.set_buffer(buffer).unwrap();
			source.set_position(source_data.position).unwrap();
			source
				.set_gain(get_gain_for_distance(source_data.distance))
				.unwrap();
			source.play();
		}

		{
			profile_scope!(profiling_frame, "wait for audio to finish");
			while source.state() == SourceState::Playing {
				std::thread::sleep(Duration::from_millis(100));
			}
		}

		source.stop();
		source.clear_buffer();

		profiling_frame.finish();
	}
}

/// Constants controlling the depth audio saliency analysis.
///
/// The depth estimate is downsampled into a coarse grid; each cell is scored by
/// how much it matters for obstacle avoidance, then the top ranked cells are
/// turned into audio sources.
const COARSE_STRIDE: usize = 4;
/// Radius (in cells) of the ring used as the local background reference when
/// detecting obstacles that protrude towards the user.
const ISOLATION_RING_RADIUS: i32 = 2;
/// Depth difference (in meters) that counts as a fully isolated obstacle.
const ISOLATION_RANGE: f32 = 1.0;
/// How strongly protruding obstacles are boosted over a flat surface.
const ISOLATION_WEIGHT: f32 = 0.6;
/// Gaussian sigma of the horizontal center bias, as a fraction of half the width.
const CENTER_SIGMA_FRACTION: f32 = 0.28;
/// Minimum Chebyshev distance (in cells) between two selected sources, so that
/// the sounds stay spread out instead of piling up on one object.
const MIN_SEPARATION: i32 = 4;
/// Saliency below this is treated as background noise (nothing interesting).
const MIN_SALIENCY: f32 = 0.02;
const AUDIBLE_DISTANCE: f32 = 6.0;
const MIN_GAIN: f32 = 0.25;
const MAX_GAIN: f32 = 1.0;
const MIN_FREQ_FACTOR: f32 = 0.8;
const MAX_FREQ_FACTOR: f32 = 1.2;

#[profile_function("profiling_frame")]
fn process_depth_estimation_data(
	depth_estimation_data: &[f32; SpatialAudioSettings::PICTURE_PIXEL_COUNT],
	segmentation_data: Option<&[i32; SpatialAudioSettings::PICTURE_PIXEL_COUNT]>,
	segmentation_class_importances: &[f32],
	settings: &SpatialAudioSettings,
	profiling_frame: &FormattedProfilingFrame,
) -> Vec<DepthAudioSourceData> {
	const GRID_W: usize = SpatialAudioSettings::PICTURE_RESOLUTION.x as usize / COARSE_STRIDE;
	const GRID_H: usize = SpatialAudioSettings::PICTURE_RESOLUTION.y as usize / COARSE_STRIDE;
	const GRID_CELLS: usize = GRID_W * GRID_H;
	const MAX_SOURCES: usize = SpatialAudioSettings::NUMBER_OF_SOURCES;

	// Downsample the depth map into a coarse grid, keeping the nearest depth and
	// the mean segmentation importance per cell.
	let mut cell_min_depth = vec![f32::MAX; GRID_CELLS];
	let mut cell_importance = vec![0.0f32; GRID_CELLS];
	for (cell_index, (cell_min_depth, cell_importance)) in cell_min_depth
		.iter_mut()
		.zip(cell_importance.iter_mut())
		.enumerate()
	{
		let cell_x = cell_index % GRID_W;
		let cell_y = cell_index / GRID_W;
		let mut importance_sum = 0.0f32;
		for y in 0..COARSE_STRIDE {
			for x in 0..COARSE_STRIDE {
				let pixel = (cell_x * COARSE_STRIDE + x)
					+ ((cell_y * COARSE_STRIDE + y)
						* SpatialAudioSettings::PICTURE_RESOLUTION.x as usize);
				let importance = segmentation_data
					.and_then(|segmentation_data| {
						segmentation_class_importances
							.get(segmentation_data[pixel] as usize)
							.copied()
					})
					.unwrap_or(1.0);
				importance_sum += importance;
				*cell_min_depth = cell_min_depth.min(depth_estimation_data[pixel]);
			}
		}
		*cell_importance = importance_sum / (COARSE_STRIDE * COARSE_STRIDE) as f32;
	}

	// Rank the cells by how relevant they are for obstacle avoidance.
	let saliency = compute_saliency_map(&cell_min_depth, &cell_importance);

	// Pick the most salient cells, spread far enough apart to remain distinguishable,
	// and fall back to the nearest cells to guarantee full angular coverage.
	let mut selected = select_salient_cells(&saliency, &cell_min_depth, MAX_SOURCES);

	// Sort by horizontal position so the sources pan from left to right in a stable
	// order instead of jumping around between frames.
	selected.sort_by_key(|&cell| cell % GRID_W);

	let mut calculate_sound_origin = CalculateSoundOrigin::new();
	let mut audio_source_data = Vec::with_capacity(selected.len());
	for cell in selected {
		let depth = cell_min_depth[cell];
		// Urgency rises towards 1.0 the closer the obstacle is.
		let proximity = 1.0 - (depth / AUDIBLE_DISTANCE).clamp(0.0, 1.0);
		let frequency = math_utils::lerp(
			proximity,
			settings.frequency * MIN_FREQ_FACTOR,
			settings.frequency * MAX_FREQ_FACTOR,
		);
		let gain = math_utils::lerp(proximity, MIN_GAIN, MAX_GAIN);

		let cell_x = cell % GRID_W;
		let cell_y = cell / GRID_W;
		let sound_origin = calculate_sound_origin.calculate_sound_origin(
			IVec2 {
				x: (cell_x * COARSE_STRIDE + COARSE_STRIDE / 2) as i32,
				y: (cell_y * COARSE_STRIDE + COARSE_STRIDE / 2) as i32,
			},
			depth,
		);

		audio_source_data.push(DepthAudioSourceData::new(
			frequency,
			settings.buffer_duration,
			SpatialAudioSettings::SAMPLE_RATE,
			sound_origin,
			gain,
			profiling_frame,
		));
	}

	audio_source_data
}

/// Scores every coarse grid cell by combining how close it is (proximity), how
/// much it sits in the walking direction (center bias), how strongly it protrudes
/// from its surroundings (isolation) and how important its segmentation class is.
fn compute_saliency_map(cell_min_depth: &[f32], cell_importance: &[f32]) -> Vec<f32> {
	const GRID_W: usize = SpatialAudioSettings::PICTURE_RESOLUTION.x as usize / COARSE_STRIDE;
	const GRID_H: usize = SpatialAudioSettings::PICTURE_RESOLUTION.y as usize / COARSE_STRIDE;

	let half_width = GRID_W as f32 / 2.0;
	let sigma = half_width * CENTER_SIGMA_FRACTION;
	let mut saliency = vec![0.0f32; GRID_W * GRID_H];

	for y in 0..GRID_H {
		for x in 0..GRID_W {
			let cell_index = x + y * GRID_W;
			let depth = cell_min_depth[cell_index];
			let proximity = 1.0 - (depth / AUDIBLE_DISTANCE).clamp(0.0, 1.0);
			let center_bias =
				(-((x as f32 - half_width + 0.5).powi(2)) / (2.0 * sigma * sigma)).exp();
			let isolation = isolation_bonus(cell_min_depth, GRID_W, GRID_H, x, y);

			saliency[cell_index] = cell_importance[cell_index]
				* (proximity * center_bias + ISOLATION_WEIGHT * isolation * proximity);
		}
	}

	saliency
}

/// How much closer this cell is than everything around it. A cell that protrudes
/// towards the user (an isolated obstacle) scores high; a flat wall/wall section
/// scores zero, which keeps it at its plain proximity weight.
fn isolation_bonus(
	cell_min_depth: &[f32],
	grid_w: usize,
	grid_h: usize,
	x: usize,
	y: usize,
) -> f32 {
	let cell_depth = cell_min_depth[x + y * grid_w];
	let mut ring_min = f32::MAX;
	for dy in -ISOLATION_RING_RADIUS..=ISOLATION_RING_RADIUS {
		for dx in -ISOLATION_RING_RADIUS..=ISOLATION_RING_RADIUS {
			if dx.abs() != ISOLATION_RING_RADIUS && dy.abs() != ISOLATION_RING_RADIUS {
				continue;
			}
			let nx = x as i32 + dx;
			let ny = y as i32 + dy;
			if nx < 0 || ny < 0 || nx >= grid_w as i32 || ny >= grid_h as i32 {
				continue;
			}
			ring_min = ring_min.min(cell_min_depth[nx as usize + ny as usize * grid_w]);
		}
	}
	if ring_min == f32::MAX {
		return 0.0;
	}
	((ring_min - cell_depth) / ISOLATION_RANGE).clamp(0.0, 1.0)
}

/// Greedily selects the ranked cells while enforcing a minimum separation, so the
/// audio sources stay spread across the field of view. Only genuinely salient
/// cells become pings. If the scene is flat and nothing crosses the saliency
/// threshold, a single ping is emitted at the nearest cell so the user still has
/// an orientation reference, but no second ping is invented.
fn select_salient_cells(
	saliency: &[f32],
	cell_min_depth: &[f32],
	max_sources: usize,
) -> Vec<usize> {
	const GRID_W: usize = SpatialAudioSettings::PICTURE_RESOLUTION.x as usize / COARSE_STRIDE;
	const GRID_H: usize = SpatialAudioSettings::PICTURE_RESOLUTION.y as usize / COARSE_STRIDE;

	let mut selected = Vec::with_capacity(max_sources);

	while selected.len() < max_sources {
		let best = best_cell(saliency, MIN_SALIENCY, GRID_W, GRID_H, &selected);
		match best {
			Some(cell) => selected.push(cell),
			None => break,
		}
	}

	if selected.is_empty() {
		let proximity: Vec<f32> = cell_min_depth
			.iter()
			.map(|&depth| 1.0 - (depth / AUDIBLE_DISTANCE).clamp(0.0, 1.0))
			.collect();
		if let Some(best) = best_cell(&proximity, 0.0, GRID_W, GRID_H, &selected) {
			selected.push(best);
		}
	}

	selected
}

/// Returns the highest scoring cell that is not within `MIN_SEPARATION` cells of an
/// already selected one, or `None` if nothing qualifies.
fn best_cell(
	scores: &[f32],
	min_score: f32,
	grid_w: usize,
	grid_h: usize,
	selected: &[usize],
) -> Option<usize> {
	let mut best: Option<(usize, f32)> = None;
	for y in 0..grid_h {
		for x in 0..grid_w {
			let cell_index = x + y * grid_w;
			let score = scores[cell_index];
			if score < min_score {
				continue;
			}
			let too_close = selected.iter().any(|&selected_cell| {
				let (selected_x, selected_y) = (selected_cell % grid_w, selected_cell / grid_w);
				(selected_x as i32 - x as i32)
					.abs()
					.max((selected_y as i32 - y as i32).abs())
					< MIN_SEPARATION
			});
			if too_close {
				continue;
			}
			if best.is_none_or(|(_, best_score)| score > best_score) {
				best = Some((cell_index, score));
			}
		}
	}
	best.map(|(cell_index, _)| cell_index)
}

#[profile_function("profiling_frame")]
fn process_object_detection_data(
	depth_estimation_data: &[f32; 256 * 256],
	object_detection_data: &[TrackedObject],
	object_label_data: &HashMap<String, ObjectLabelData>,
	profiling_frame: &FormattedProfilingFrame,
) -> VecDeque<ObjectAudioSourceData> {
	let mut audio_source_data = VecDeque::new();

	for tracked_object in object_detection_data {
		let object = &tracked_object.object;

		let object_name = object.class_name.to_lowercase().trim().to_owned();

		let Some(object_label_data) = object_label_data.get(&object_name) else {
			error!(
				"[ProcessObjectDetectionData] Could not find object {} in the object_label_data. Skipping to next one ...",
				object_name
			);
			continue;
		};

		let coord = IVec2 {
			x: (object.bbox.center_x * (SpatialAudioSettings::PICTURE_RESOLUTION.x as f32 - 1.0))
				as i32,
			y: (object.bbox.center_y * (SpatialAudioSettings::PICTURE_RESOLUTION.y as f32 - 1.0))
				as i32,
		};

		trace!(
			"Object {}: Start: {} End: {}",
			object_name, object_label_data.sample_begin, object_label_data.sample_end
		);

		let distance = depth_estimation_data
			[(coord.x + (coord.y * SpatialAudioSettings::PICTURE_RESOLUTION.x)) as usize];
		let mut calculate_sound_origin = CalculateSoundOrigin::new();
		let sound_origin = calculate_sound_origin.calculate_sound_origin(coord, distance);

		audio_source_data.push_back(ObjectAudioSourceData {
			// TODO: object.tracking_id needs to be provided by object tracking, for now class id works as well
			object_id: object.class_id,
			name: object_name,
			sound_begin: object_label_data.sample_begin,
			sound_end: object_label_data.sample_end,
			position: sound_origin,
			distance,
		});
	}

	audio_source_data
}

fn get_gain_for_distance(distance: f32) -> f32 {
	const CLOSEST_DISTANCE: f32 = 1.0;
	const FARTHEST_DISTANCE: f32 = 3.5;
	const GAIN_CLOSE: f32 = 1.0;
	const GAIN_FAR: f32 = 0.2;

	math_utils::remap(
		distance,
		(CLOSEST_DISTANCE, FARTHEST_DISTANCE),
		(GAIN_CLOSE, GAIN_FAR),
	)
}

/// <https://gist.github.com/laundmo/cb06630109e5e1100f5a2758dfb67cfd>
mod math_utils {
	#[inline(always)]
	pub fn lerp(value: f32, from: f32, to: f32) -> f32 {
		(1.0 - value) * from + value * to
	}
	#[inline(always)]
	pub fn inv_lerp(value: f32, from: f32, to: f32) -> f32 {
		(value - from) / (to - from)
	}
	#[inline(always)]
	pub fn remap(value: f32, from_range: (f32, f32), to_range: (f32, f32)) -> f32 {
		lerp(
			inv_lerp(value, from_range.0, from_range.1),
			to_range.0,
			to_range.1,
		)
	}
}
