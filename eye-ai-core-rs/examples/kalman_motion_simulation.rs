//! Deterministic, isolated comparison of EyeAI's real ByteTrack/Kalman path.
//!
//! Run from `eye-ai-core-rs` with:
//! `cargo run --release --offline --features kalman-motion-benchmark --example kalman_motion_simulation`
//!
//! The harness writes raw per-run metrics and a generated summary below
//! `target/kalman-motion-simulation/`. It never changes production settings.

use bytetrack_cpp_rs::{BYTETracker, Object, Rect, STrack};
use std::{
	collections::{BTreeMap, HashMap, HashSet},
	f32::consts::TAU,
	fmt::Write as _,
	fs,
	path::PathBuf,
	time::Duration,
};

const DURATION_SECONDS: f32 = 8.0;
const SEEDS: [u64; 6] = [7, 19, 43, 101, 271, 811];
const VALIDATION_SECONDS: f32 = 0.45;
const MAX_TRACKING_TIME_SECONDS: f32 = 10.0;
const ASSIGNMENT_DISTANCE: f32 = 0.18;

#[derive(Clone, Copy, Debug)]
struct BBox {
	cx: f32,
	cy: f32,
	w: f32,
	h: f32,
}

impl BBox {
	fn distance(self, other: Self) -> f32 {
		(self.cx - other.cx).hypot(self.cy - other.cy)
	}

	fn to_object(self, confidence: f32) -> Object {
		Object {
			rect: Rect::new(
				self.cx - self.w / 2.0,
				self.cy - self.h / 2.0,
				self.w,
				self.h,
			),
			label: 0,
			prob: confidence,
		}
	}

	fn from_track(track: &STrack) -> Self {
		Self {
			cx: track.rect.x + track.rect.width / 2.0,
			cy: track.rect.y + track.rect.height / 2.0,
			w: track.rect.width,
			h: track.rect.height,
		}
	}
}

#[derive(Clone, Debug)]
struct Truth {
	id: usize,
	bbox: BBox,
	in_frame: bool,
	detected: bool,
}

#[derive(Clone, Debug)]
struct Frame {
	dt: Duration,
	time: f32,
	truths: Vec<Truth>,
	detections: Vec<Object>,
	camera_motion_score: f32,
	body_motion_score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Scenario {
	StillConstant,
	StillManeuver,
	SlowPan,
	FastPan,
	AbruptCameraReverse,
	WalkingJitter,
	CameraAndObject,
	PeopleDuringPan,
	CrossingTracks,
	ShortOcclusion,
	LeavesImage,
	CombinedStress,
	WalkingRealistic,
}

impl Scenario {
	const ALL: [Self; 13] = [
		Self::StillConstant,
		Self::StillManeuver,
		Self::SlowPan,
		Self::FastPan,
		Self::AbruptCameraReverse,
		Self::WalkingJitter,
		Self::CameraAndObject,
		Self::PeopleDuringPan,
		Self::CrossingTracks,
		Self::ShortOcclusion,
		Self::LeavesImage,
		Self::CombinedStress,
		// Keep this new scenario last so all existing scenario-derived seeds and
		// the original aggressive Walking/Jitter case remain unchanged.
		Self::WalkingRealistic,
	];

	fn name(self) -> &'static str {
		match self {
			Self::StillConstant => "01_still_constant_object",
			Self::StillManeuver => "02_still_acceleration_and_turn",
			Self::SlowPan => "03_slow_camera_pan",
			Self::FastPan => "04_fast_camera_pan",
			Self::AbruptCameraReverse => "05_abrupt_camera_reverse",
			Self::WalkingJitter => "06_walking_camera_jitter",
			Self::CameraAndObject => "07_camera_and_object_motion",
			Self::PeopleDuringPan => "08_people_during_pan",
			Self::CrossingTracks => "09_crossing_tracks",
			Self::ShortOcclusion => "10_short_detection_occlusion",
			Self::LeavesImage => "11_object_leaves_image",
			Self::CombinedStress => "12_combined_noise_dropouts_motion",
			Self::WalkingRealistic => "WALKING_REALISTIC",
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Cadence {
	Hz5,
	Hz10,
	Hz15,
	Hz30,
	Variable,
}

impl Cadence {
	const ALL: [Self; 5] = [
		Self::Hz5,
		Self::Hz10,
		Self::Hz15,
		Self::Hz30,
		Self::Variable,
	];

	fn name(self) -> &'static str {
		match self {
			Self::Hz5 => "5_hz",
			Self::Hz10 => "10_hz",
			Self::Hz15 => "15_hz",
			Self::Hz30 => "30_hz",
			Self::Variable => "variable",
		}
	}

	fn intervals(self) -> Vec<Duration> {
		let fixed_hz: Option<f32> = match self {
			Self::Hz5 => Some(5.0),
			Self::Hz10 => Some(10.0),
			Self::Hz15 => Some(15.0),
			Self::Hz30 => Some(30.0),
			Self::Variable => None,
		};
		let pattern = [1.0 / 15.0, 1.0 / 8.0, 1.0 / 20.0, 1.0 / 5.0, 1.0 / 12.0];
		let mut intervals = Vec::new();
		let mut elapsed = 0.0_f32;
		let mut index = 0;
		while elapsed < DURATION_SECONDS {
			let seconds = fixed_hz.map_or(pattern[index % pattern.len()], |hz| 1.0 / hz);
			let remaining = DURATION_SECONDS - elapsed;
			let seconds = seconds.min(remaining);
			intervals.push(Duration::from_secs_f32(seconds));
			elapsed += seconds;
			index += 1;
		}
		intervals
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum PhoneProfile {
	StrongCorrelation,
	NoisyCorrelation,
	Delayed,
	FalsePositive,
	FalseNegative,
	WalkingMismatch,
}

impl PhoneProfile {
	const ALL: [Self; 6] = [
		Self::StrongCorrelation,
		Self::NoisyCorrelation,
		Self::Delayed,
		Self::FalsePositive,
		Self::FalseNegative,
		Self::WalkingMismatch,
	];

	fn name(self) -> &'static str {
		match self {
			Self::StrongCorrelation => "strong_correlation",
			Self::NoisyCorrelation => "correlated_with_noise",
			Self::Delayed => "delayed_250ms",
			Self::FalsePositive => "false_positive",
			Self::FalseNegative => "false_negative",
			Self::WalkingMismatch => "walking_body_mismatch",
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Variant {
	NoPrediction,
	CurrentKalman,
	MotionAware { strength: f32 },
}

impl Variant {
	fn name(self) -> &'static str {
		match self {
			Self::NoPrediction => "A_no_prediction",
			Self::CurrentKalman => "B_current_kalman",
			Self::MotionAware { strength } if strength == 0.5 => "C_weak",
			Self::MotionAware { strength } if strength == 1.0 => "C_medium",
			Self::MotionAware { .. } => "C_strong",
		}
	}

	fn prediction_enabled(self) -> bool {
		!matches!(self, Self::NoPrediction)
	}

	fn q_scale(self, phone_motion_score: f32) -> f32 {
		match self {
			Self::MotionAware { strength } => (1.0 + strength * phone_motion_score).clamp(1.0, 3.0),
			_ => 1.0,
		}
	}
}

#[derive(Clone)]
struct DeterministicRng {
	state: u64,
}

impl DeterministicRng {
	fn new(seed: u64) -> Self {
		Self { state: seed.max(1) }
	}

	fn uniform(&mut self) -> f32 {
		self.state ^= self.state << 13;
		self.state ^= self.state >> 7;
		self.state ^= self.state << 17;
		((self.state >> 40) as f32 + 0.5) / ((1_u32 << 24) as f32)
	}

	fn normal(&mut self) -> f32 {
		let u1 = self.uniform().max(1e-7);
		let u2 = self.uniform();
		(-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
	}
}

fn camera_position(scenario: Scenario, t: f32) -> f32 {
	match scenario {
		Scenario::StillConstant
		| Scenario::StillManeuver
		| Scenario::CrossingTracks
		| Scenario::ShortOcclusion
		| Scenario::LeavesImage => 0.0,
		Scenario::SlowPan => 0.10 * t,
		Scenario::FastPan => 0.31 * t,
		Scenario::AbruptCameraReverse => {
			if t <= 3.2 {
				0.26 * t
			} else {
				0.26 * 3.2 - 0.34 * (t - 3.2)
			}
		}
		Scenario::WalkingJitter => 0.075 * (TAU * 1.55 * t).sin() + 0.025 * (TAU * 4.1 * t).sin(),
		// Conservative walking-camera model: a coherent linear component with
		// small lateral sway and very small high-frequency hand/foot jitter.
		// The linear term remains larger than the combined oscillatory velocity,
		// so horizontal image motion stays predominantly one-directional.
		Scenario::WalkingRealistic => {
			-0.18 + 0.060 * t + 0.008 * (TAU * 0.65 * t).sin() + 0.0008 * (TAU * 4.5 * t).sin()
		}
		Scenario::CameraAndObject => 0.16 * t + 0.025 * (TAU * 1.2 * t).sin(),
		Scenario::PeopleDuringPan => 0.12 * t + 0.035 * (TAU * 0.35 * t).sin(),
		Scenario::CombinedStress => {
			0.13 * t + 0.055 * (TAU * 1.15 * t).sin() + 0.018 * (TAU * 4.7 * t).sin()
		}
	}
}

fn scene_boxes(scenario: Scenario, t: f32) -> Vec<BBox> {
	let camera = camera_position(scenario, t);
	let standard = |world_x: f32, y: f32| BBox {
		cx: world_x - camera,
		cy: y,
		w: 0.15,
		h: 0.24,
	};
	match scenario {
		Scenario::StillConstant => vec![standard(0.20 + 0.075 * t, 0.52)],
		Scenario::StillManeuver => {
			let x = if t < 2.5 {
				0.18 + 0.045 * t + 0.018 * t * t
			} else if t < 4.5 {
				0.405 + 0.16 * (t - 2.5)
			} else {
				0.725 - 0.19 * (t - 4.5)
			};
			vec![standard(x, 0.52)]
		}
		Scenario::SlowPan => vec![standard(0.82, 0.52)],
		Scenario::FastPan => vec![standard(1.55, 0.52)],
		Scenario::AbruptCameraReverse => vec![standard(1.15, 0.52)],
		Scenario::WalkingJitter => vec![standard(0.50, 0.52)],
		Scenario::WalkingRealistic => {
			let sway_phase = TAU * 0.65 * t;
			let bobbing = 0.012 * (TAU * 1.20 * t).sin() + 0.002 * (TAU * 3.80 * t).sin();
			let scale = 1.0 + 0.030 * (sway_phase + 0.40).sin();
			vec![BBox {
				cx: 0.50 - camera,
				cy: 0.52 + bobbing,
				w: 0.15 * scale,
				h: 0.24 * (1.0 + 0.035 * sway_phase.sin()),
			}]
		}
		Scenario::CameraAndObject => vec![standard(0.35 + 0.255 * t, 0.50)],
		Scenario::PeopleDuringPan => [0.55, 0.82, 1.09, 1.36]
			.into_iter()
			.enumerate()
			.map(|(index, x)| {
				standard(
					x + 0.012 * (TAU * (0.25 + index as f32 * 0.04) * t).sin(),
					0.30 + index as f32 * 0.14,
				)
			})
			.collect(),
		Scenario::CrossingTracks => vec![
			BBox {
				cx: 0.18 + 0.09 * t,
				cy: 0.47,
				w: 0.14,
				h: 0.23,
			},
			BBox {
				cx: 0.82 - 0.09 * t,
				cy: 0.53,
				w: 0.14,
				h: 0.23,
			},
		],
		Scenario::ShortOcclusion => vec![standard(0.18 + 0.105 * t, 0.52)],
		Scenario::LeavesImage => vec![standard(0.15 + 0.145 * t, 0.52)],
		Scenario::CombinedStress => vec![
			standard(0.35 + 0.23 * t + 0.035 * (TAU * 0.7 * t).sin(), 0.40),
			standard(1.00 + 0.06 * t - 0.04 * (TAU * 0.55 * t).sin(), 0.64),
		],
	}
}

fn forced_dropout(scenario: Scenario, object_id: usize, t: f32) -> bool {
	match scenario {
		Scenario::FastPan => (3.00..3.42).contains(&t),
		Scenario::WalkingJitter => (4.10..4.36).contains(&t),
		Scenario::WalkingRealistic => (4.10..4.36).contains(&t),
		Scenario::ShortOcclusion => (2.35..2.82).contains(&t) || (5.10..5.43).contains(&t),
		Scenario::PeopleDuringPan => object_id == 1 && (3.00..3.46).contains(&t),
		Scenario::CrossingTracks => object_id == 0 && (3.25..3.52).contains(&t),
		Scenario::CombinedStress => {
			(object_id == 0 && (2.75..3.28).contains(&t))
				|| (object_id == 1 && (5.05..5.42).contains(&t))
		}
		_ => false,
	}
}

fn generate_sequence(scenario: Scenario, cadence: Cadence, seed: u64) -> Vec<Frame> {
	let mut rng =
		DeterministicRng::new(seed ^ ((scenario as u64 + 1) << 20) ^ ((cadence as u64 + 1) << 32));
	let mut time = 0.0_f32;
	let mut previous_camera = camera_position(scenario, 0.0);
	let mut previous_velocity = 0.0_f32;
	let mut frames = Vec::new();

	for dt in cadence.intervals() {
		let dt_seconds = dt.as_secs_f32();
		time += dt_seconds;
		let camera = camera_position(scenario, time);
		let camera_velocity = (camera - previous_camera) / dt_seconds.max(1e-6);
		let camera_acceleration = (camera_velocity - previous_velocity) / dt_seconds.max(1e-6);
		let camera_motion_score =
			(camera_velocity.abs() / 0.40 + camera_acceleration.abs() / 18.0).clamp(0.0, 1.0);
		let body_motion_score = (0.34
			+ 0.28 * (TAU * 1.8 * time + seed as f32 * 0.01).sin()
			+ 0.13 * (TAU * 3.7 * time).sin())
		.clamp(0.0, 1.0);
		previous_camera = camera;
		previous_velocity = camera_velocity;

		let boxes = scene_boxes(scenario, time);
		let noise_sigma = if scenario == Scenario::CombinedStress {
			0.018
		} else {
			0.008
		};
		let confidence_sigma = if scenario == Scenario::CombinedStress {
			0.13
		} else {
			0.07
		};
		let random_false_negative = if scenario == Scenario::CombinedStress {
			0.07
		} else {
			0.018
		};
		let mut truths = Vec::with_capacity(boxes.len());
		let mut detections = Vec::with_capacity(boxes.len() + 1);

		for (id, bbox) in boxes.into_iter().enumerate() {
			let in_frame = bbox.cx + bbox.w / 2.0 > 0.0
				&& bbox.cx - bbox.w / 2.0 < 1.0
				&& bbox.cy + bbox.h / 2.0 > 0.0
				&& bbox.cy - bbox.h / 2.0 < 1.0;
			let detected = in_frame
				&& !forced_dropout(scenario, id, time)
				&& rng.uniform() >= random_false_negative;
			truths.push(Truth {
				id,
				bbox,
				in_frame,
				detected,
			});
			if detected {
				let noisy_bbox = BBox {
					cx: bbox.cx + noise_sigma * rng.normal(),
					cy: bbox.cy + noise_sigma * rng.normal(),
					w: (bbox.w * (1.0 + 0.055 * rng.normal())).max(0.04),
					h: (bbox.h * (1.0 + 0.045 * rng.normal())).max(0.06),
				};
				let confidence = (0.86 + confidence_sigma * rng.normal()).clamp(0.50, 0.99);
				detections.push(noisy_bbox.to_object(confidence));
			}
		}

		let false_positive_probability = match scenario {
			Scenario::CombinedStress => 0.08,
			Scenario::PeopleDuringPan => 0.025,
			_ => 0.006,
		};
		if rng.uniform() < false_positive_probability {
			let false_box = BBox {
				cx: 0.05 + 0.90 * rng.uniform(),
				cy: 0.12 + 0.76 * rng.uniform(),
				w: 0.08 + 0.10 * rng.uniform(),
				h: 0.12 + 0.18 * rng.uniform(),
			};
			detections.push(false_box.to_object(0.61 + 0.25 * rng.uniform()));
		}

		frames.push(Frame {
			dt,
			time,
			truths,
			detections,
			camera_motion_score,
			body_motion_score,
		});
	}
	frames
}

fn phone_motion_scores(frames: &[Frame], profile: PhoneProfile, seed: u64) -> Vec<f32> {
	let mut rng = DeterministicRng::new(seed ^ ((profile as u64 + 17) << 36));
	let mut scores = Vec::with_capacity(frames.len());
	for (index, frame) in frames.iter().enumerate() {
		let camera = frame.camera_motion_score;
		let score = match profile {
			PhoneProfile::StrongCorrelation => {
				0.88 * camera + 0.08 * frame.body_motion_score + 0.025 * rng.normal()
			}
			PhoneProfile::NoisyCorrelation => {
				0.67 * camera + 0.18 * frame.body_motion_score + 0.14 * rng.normal()
			}
			PhoneProfile::Delayed => {
				let target_time = frame.time - 0.25;
				let delayed = frames[..=index]
					.iter()
					.rev()
					.find(|candidate| candidate.time <= target_time)
					.map_or(0.0, |candidate| candidate.camera_motion_score);
				0.86 * delayed + 0.10 * frame.body_motion_score
			}
			PhoneProfile::FalsePositive => {
				0.76 + 0.16 * (TAU * 1.4 * frame.time).sin() + 0.06 * rng.normal()
			}
			PhoneProfile::FalseNegative => {
				0.035 + 0.06 * frame.body_motion_score + 0.025 * rng.normal()
			}
			PhoneProfile::WalkingMismatch => {
				0.35 * camera + 0.56 * frame.body_motion_score + 0.07 * rng.normal()
			}
		};
		scores.push(score.clamp(0.0, 1.0));
	}
	scores
}

#[derive(Clone, Copy, Debug)]
struct Validation {
	confidence_visible_seconds: f32,
	confirmed: bool,
	last_seen_time: f32,
	last_seen_update: u64,
	first_seen_time: f32,
}

#[derive(Default)]
struct ValidationGate {
	update_number: u64,
	states: HashMap<i32, Validation>,
}

impl ValidationGate {
	fn update(&mut self, tracks: &[STrack], now: f32, dt: Duration) -> Vec<(i32, f32)> {
		self.update_number = self.update_number.wrapping_add(1);
		let mut newly_confirmed = Vec::new();
		for track in tracks {
			let state = self.states.entry(track.track_id).or_insert(Validation {
				confidence_visible_seconds: 0.0,
				confirmed: false,
				last_seen_time: now,
				last_seen_update: self.update_number,
				first_seen_time: now,
			});
			let consecutive = state.last_seen_update == self.update_number.wrapping_sub(1);
			state.last_seen_time = now;
			state.last_seen_update = self.update_number;
			if !state.confirmed && consecutive && dt.as_secs_f32() <= VALIDATION_SECONDS {
				let confidence = if track.score.is_finite() {
					track.score.clamp(0.0, 1.0)
				} else {
					0.0
				};
				state.confidence_visible_seconds += confidence * dt.as_secs_f32();
				if state.confidence_visible_seconds >= VALIDATION_SECONDS {
					state.confirmed = true;
					newly_confirmed.push((track.track_id, now - state.first_seen_time));
				}
			}
		}
		self.states
			.retain(|_, state| now - state.last_seen_time <= MAX_TRACKING_TIME_SECONDS);
		newly_confirmed
	}

	fn is_confirmed(&self, id: i32) -> bool {
		self.states.get(&id).is_some_and(|state| state.confirmed)
	}
}

#[derive(Default, Clone, Debug)]
struct Metrics {
	id_switches: u64,
	fragments: u64,
	new_ids: u64,
	false_associations: u64,
	lost_reassociation_opportunities: u64,
	lost_reassociation_successes: u64,
	short_dropout_opportunities: u64,
	short_dropout_successes: u64,
	new_tentative_after_loss: u64,
	false_positive_tracks: u64,
	in_frame_truth_updates: u64,
	detected_truth_updates: u64,
	assigned_truth_updates: u64,
	confirmed_truth_updates: u64,
	raw_track_observations: u64,
	confirmed_track_observations: u64,
	position_errors: Vec<f32>,
	confirmation_delays: Vec<f32>,
}

#[derive(Default)]
struct TruthHistory {
	ever_assigned: bool,
	missing: bool,
	last_id: Option<i32>,
	gap_prior_id: Option<i32>,
	detection_gap_active: bool,
	pre_dropout_id: Option<i32>,
	ids: HashSet<i32>,
}

fn assign_tracks(tracks: &[STrack], truths: &[Truth]) -> Vec<Option<usize>> {
	let mut assignments = vec![None; truths.len()];
	let mut pairs = Vec::new();
	for (truth_index, truth) in truths
		.iter()
		.enumerate()
		.filter(|(_, truth)| truth.in_frame)
	{
		for (track_index, track) in tracks.iter().enumerate() {
			pairs.push((
				BBox::from_track(track).distance(truth.bbox),
				truth_index,
				track_index,
			));
		}
	}
	pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
	let mut used_tracks = vec![false; tracks.len()];
	for (distance, truth_index, track_index) in pairs {
		if distance > ASSIGNMENT_DISTANCE
			|| assignments[truth_index].is_some()
			|| used_tracks[track_index]
		{
			continue;
		}
		assignments[truth_index] = Some(track_index);
		used_tracks[track_index] = true;
	}
	assignments
}

fn run_variant(frames: &[Frame], variant: Variant, phone_scores: &[f32]) -> Metrics {
	assert_eq!(frames.len(), phone_scores.len());
	let mut tracker = BYTETracker::default();
	let mut validation = ValidationGate::default();
	let mut histories: HashMap<usize, TruthHistory> = HashMap::new();
	let mut id_owner: HashMap<i32, usize> = HashMap::new();
	let mut recorded_confirmation_ids = HashSet::new();
	let mut metrics = Metrics::default();

	for (frame_index, (frame, &phone_score)) in frames.iter().zip(phone_scores).enumerate() {
		let elapsed = if frame_index == 0 {
			Duration::ZERO
		} else {
			frame.dt
		};
		let tracks = tracker.update_for_benchmark(
			&frame.detections,
			elapsed,
			variant.prediction_enabled(),
			variant.q_scale(phone_score),
		);
		let newly_confirmed = validation.update(&tracks, frame.time, elapsed);
		let newly_confirmed = newly_confirmed.into_iter().collect::<HashMap<_, _>>();
		metrics.raw_track_observations += tracks.len() as u64;
		metrics.confirmed_track_observations += tracks
			.iter()
			.filter(|track| validation.is_confirmed(track.track_id))
			.count() as u64;

		let assignments = assign_tracks(&tracks, &frame.truths);
		let assigned_track_indices = assignments
			.iter()
			.flatten()
			.copied()
			.collect::<HashSet<_>>();
		metrics.false_positive_tracks +=
			tracks.len().saturating_sub(assigned_track_indices.len()) as u64;

		for (truth_index, truth) in frame.truths.iter().enumerate() {
			if !truth.in_frame {
				continue;
			}
			metrics.in_frame_truth_updates += 1;
			if truth.detected {
				metrics.detected_truth_updates += 1;
			}
			let history = histories.entry(truth.id).or_default();
			if !truth.detected && history.ever_assigned && !history.detection_gap_active {
				history.detection_gap_active = true;
				history.pre_dropout_id = history.last_id;
			}
			let assigned_id =
				assignments[truth_index].map(|track_index| tracks[track_index].track_id);
			if truth.detected && history.detection_gap_active {
				metrics.short_dropout_opportunities += 1;
				if assigned_id.is_some() && assigned_id == history.pre_dropout_id {
					metrics.short_dropout_successes += 1;
				}
				history.detection_gap_active = false;
				history.pre_dropout_id = None;
			}
			let Some(track_index) = assignments[truth_index] else {
				if history.ever_assigned && !history.missing {
					history.gap_prior_id = history.last_id;
				}
				if history.ever_assigned {
					history.missing = true;
				}
				continue;
			};

			let track = &tracks[track_index];
			let id = track.track_id;
			metrics.assigned_truth_updates += 1;
			if validation.is_confirmed(id) {
				metrics.confirmed_truth_updates += 1;
			}
			metrics
				.position_errors
				.push(BBox::from_track(track).distance(truth.bbox));
			if let Some(owner) = id_owner.get(&id) {
				if *owner != truth.id {
					metrics.false_associations += 1;
				}
			} else {
				id_owner.insert(id, truth.id);
			}

			let is_new_id_for_truth = !history.ids.contains(&id);
			if history.ever_assigned {
				if history.last_id != Some(id) {
					metrics.id_switches += 1;
				}
				if history.missing {
					metrics.fragments += 1;
					metrics.lost_reassociation_opportunities += 1;
					if history.gap_prior_id == Some(id) {
						metrics.lost_reassociation_successes += 1;
					}
					if is_new_id_for_truth && !validation.is_confirmed(id) {
						metrics.new_tentative_after_loss += 1;
					}
				}
			}
			if is_new_id_for_truth && !history.ids.is_empty() {
				metrics.new_ids += 1;
			}
			history.ids.insert(id);
			history.ever_assigned = true;
			history.missing = false;
			history.last_id = Some(id);
			history.gap_prior_id = None;

			if validation.is_confirmed(id) && recorded_confirmation_ids.insert(id) {
				if let Some(delay) = newly_confirmed.get(&id) {
					metrics.confirmation_delays.push(*delay);
				}
			}
		}
	}
	metrics
}

#[derive(Clone, Debug)]
struct RunResult {
	scenario: Scenario,
	cadence: Cadence,
	seed: u64,
	variant: Variant,
	phone_profile: Option<PhoneProfile>,
	metrics: Metrics,
}

fn percentile(values: &[f32], quantile: f32) -> f32 {
	if values.is_empty() {
		return 0.0;
	}
	let mut sorted = values.to_vec();
	sorted.sort_by(f32::total_cmp);
	let index = ((sorted.len() - 1) as f32 * quantile).round() as usize;
	sorted[index]
}

fn mean(values: &[f32]) -> f32 {
	if values.is_empty() {
		0.0
	} else {
		values.iter().sum::<f32>() / values.len() as f32
	}
}

#[derive(Default, Debug)]
struct Aggregate {
	runs: usize,
	id_switches: u64,
	fragments: u64,
	new_ids: u64,
	false_associations: u64,
	reassoc_opportunities: u64,
	reassoc_successes: u64,
	short_dropout_opportunities: u64,
	short_dropout_successes: u64,
	new_tentative_after_loss: u64,
	false_positive_tracks: u64,
	in_frame_truth_updates: u64,
	detected_truth_updates: u64,
	assigned_truth_updates: u64,
	confirmed_truth_updates: u64,
	raw_track_observations: u64,
	confirmed_track_observations: u64,
	position_errors: Vec<f32>,
	confirmation_delays: Vec<f32>,
	worst_run_id_switches: u64,
	worst_run_fragments: u64,
}

impl Aggregate {
	fn add(&mut self, metrics: &Metrics) {
		self.runs += 1;
		self.id_switches += metrics.id_switches;
		self.fragments += metrics.fragments;
		self.new_ids += metrics.new_ids;
		self.false_associations += metrics.false_associations;
		self.reassoc_opportunities += metrics.lost_reassociation_opportunities;
		self.reassoc_successes += metrics.lost_reassociation_successes;
		self.short_dropout_opportunities += metrics.short_dropout_opportunities;
		self.short_dropout_successes += metrics.short_dropout_successes;
		self.new_tentative_after_loss += metrics.new_tentative_after_loss;
		self.false_positive_tracks += metrics.false_positive_tracks;
		self.in_frame_truth_updates += metrics.in_frame_truth_updates;
		self.detected_truth_updates += metrics.detected_truth_updates;
		self.assigned_truth_updates += metrics.assigned_truth_updates;
		self.confirmed_truth_updates += metrics.confirmed_truth_updates;
		self.raw_track_observations += metrics.raw_track_observations;
		self.confirmed_track_observations += metrics.confirmed_track_observations;
		self.position_errors
			.extend_from_slice(&metrics.position_errors);
		self.confirmation_delays
			.extend_from_slice(&metrics.confirmation_delays);
		self.worst_run_id_switches = self.worst_run_id_switches.max(metrics.id_switches);
		self.worst_run_fragments = self.worst_run_fragments.max(metrics.fragments);
	}

	fn reassociation_rate(&self) -> f32 {
		if self.reassoc_opportunities == 0 {
			0.0
		} else {
			100.0 * self.reassoc_successes as f32 / self.reassoc_opportunities as f32
		}
	}

	fn confirmation_mean(&self) -> f32 {
		mean(&self.confirmation_delays)
	}

	fn percentage(numerator: u64, denominator: u64) -> f32 {
		if denominator == 0 {
			0.0
		} else {
			100.0 * numerator as f32 / denominator as f32
		}
	}
}

fn aggregate<'a>(results: impl Iterator<Item = &'a RunResult>) -> Aggregate {
	let mut aggregate = Aggregate::default();
	for result in results {
		aggregate.add(&result.metrics);
	}
	aggregate
}

fn assert_production_baseline_equivalence() {
	let mut production = BYTETracker::default();
	let mut benchmark = BYTETracker::default();
	for index in 0..24 {
		let dt = if index == 0 {
			Duration::ZERO
		} else {
			Duration::from_millis(73 + (index % 4) * 11)
		};
		let objects = if matches!(index, 9 | 10 | 17) {
			Vec::new()
		} else {
			vec![
				BBox {
					cx: 0.20 + index as f32 * 0.018,
					cy: 0.5,
					w: 0.15,
					h: 0.24,
				}
				.to_object(0.87),
			]
		};
		let normal = production.update(&objects, dt);
		let baseline = benchmark.update_for_benchmark(&objects, dt, true, 1.0);
		assert_eq!(
			normal, baseline,
			"benchmark baseline diverged at update {index}"
		);
	}
}

fn run_simulation() -> Vec<RunResult> {
	let mut results = Vec::new();
	for scenario in Scenario::ALL {
		for cadence in Cadence::ALL {
			for seed in SEEDS {
				let frames = generate_sequence(scenario, cadence, seed);
				let neutral_phone = vec![0.0; frames.len()];
				for variant in [Variant::NoPrediction, Variant::CurrentKalman] {
					results.push(RunResult {
						scenario,
						cadence,
						seed,
						variant,
						phone_profile: None,
						metrics: run_variant(&frames, variant, &neutral_phone),
					});
				}
				for phone_profile in PhoneProfile::ALL {
					let phone_scores = phone_motion_scores(&frames, phone_profile, seed);
					for strength in [0.5, 1.0, 2.0] {
						let variant = Variant::MotionAware { strength };
						results.push(RunResult {
							scenario,
							cadence,
							seed,
							variant,
							phone_profile: Some(phone_profile),
							metrics: run_variant(&frames, variant, &phone_scores),
						});
					}
				}
			}
		}
	}
	results
}

fn write_csv(results: &[RunResult], output_dir: &PathBuf) -> std::io::Result<()> {
	let mut csv = String::from(
		"scenario,cadence,seed,variant,phone_profile,id_switches,fragments,new_ids,false_associations,reassoc_opportunities,reassoc_successes,reassoc_rate_percent,short_dropout_opportunities,short_dropout_successes,short_dropout_survival_percent,new_tentative_after_loss,false_positive_tracks,in_frame_truth_updates,detected_truth_updates,assigned_truth_updates,track_coverage_percent,confirmed_truth_updates,confirmed_coverage_percent,raw_track_observations,confirmed_track_observations,mean_position_error,p95_position_error,max_position_error,mean_confirmation_seconds,p95_confirmation_seconds\n",
	);
	for result in results {
		let m = &result.metrics;
		let reassoc_rate = if m.lost_reassociation_opportunities == 0 {
			0.0
		} else {
			100.0 * m.lost_reassociation_successes as f32
				/ m.lost_reassociation_opportunities as f32
		};
		let dropout_rate =
			Aggregate::percentage(m.short_dropout_successes, m.short_dropout_opportunities);
		writeln!(
			csv,
			"{},{},{},{},{},{},{},{},{},{},{},{:.3},{},{},{:.3},{},{},{},{},{},{:.3},{},{:.3},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
			result.scenario.name(),
			result.cadence.name(),
			result.seed,
			result.variant.name(),
			result.phone_profile.map_or("none", PhoneProfile::name),
			m.id_switches,
			m.fragments,
			m.new_ids,
			m.false_associations,
			m.lost_reassociation_opportunities,
			m.lost_reassociation_successes,
			reassoc_rate,
			m.short_dropout_opportunities,
			m.short_dropout_successes,
			dropout_rate,
			m.new_tentative_after_loss,
			m.false_positive_tracks,
			m.in_frame_truth_updates,
			m.detected_truth_updates,
			m.assigned_truth_updates,
			Aggregate::percentage(m.assigned_truth_updates, m.in_frame_truth_updates),
			m.confirmed_truth_updates,
			Aggregate::percentage(m.confirmed_truth_updates, m.in_frame_truth_updates),
			m.raw_track_observations,
			m.confirmed_track_observations,
			mean(&m.position_errors),
			percentile(&m.position_errors, 0.95),
			m.position_errors.iter().copied().fold(0.0_f32, f32::max),
			mean(&m.confirmation_delays),
			percentile(&m.confirmation_delays, 0.95),
		)
		.expect("writing to String cannot fail");
	}
	fs::write(output_dir.join("raw_results.csv"), csv)
}

fn table_row(label: &str, aggregate: &Aggregate) -> String {
	let per_run = |value: u64| value as f32 / aggregate.runs.max(1) as f32;
	format!(
		"| {label} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1}% | {:.1}% | {:.1}% | {:.1}% ({}/{}) | {:.3} | {:.4} | {:.4} | {:.3} | {} / {} |\n",
		per_run(aggregate.id_switches),
		per_run(aggregate.fragments),
		per_run(aggregate.new_ids),
		per_run(aggregate.false_associations),
		Aggregate::percentage(
			aggregate.assigned_truth_updates,
			aggregate.in_frame_truth_updates
		),
		Aggregate::percentage(
			aggregate.confirmed_truth_updates,
			aggregate.in_frame_truth_updates
		),
		aggregate.reassociation_rate(),
		Aggregate::percentage(
			aggregate.short_dropout_successes,
			aggregate.short_dropout_opportunities,
		),
		aggregate.short_dropout_successes,
		aggregate.short_dropout_opportunities,
		per_run(aggregate.new_tentative_after_loss),
		mean(&aggregate.position_errors),
		percentile(&aggregate.position_errors, 0.95),
		aggregate.confirmation_mean(),
		aggregate.worst_run_id_switches,
		aggregate.worst_run_fragments,
	)
}

fn generated_summary(results: &[RunResult]) -> String {
	let mut report = String::new();
	report.push_str("# Generated Kalman motion simulation summary\n\n");
	writeln!(
		report,
		"Deterministic run matrix: {} scenarios × {} cadences × {} seeds. A and B each have {} runs; each C strength has {} runs across six PhoneMotion profiles. Position units are normalized image coordinates.\n",
		Scenario::ALL.len(),
		Cadence::ALL.len(),
		SEEDS.len(),
		Scenario::ALL.len() * Cadence::ALL.len() * SEEDS.len(),
		Scenario::ALL.len() * Cadence::ALL.len() * SEEDS.len() * PhoneProfile::ALL.len(),
	)
	.expect("writing to String cannot fail");
	report.push_str("Event counts are means per run; rates and position errors are pooled.\n\n");
	report.push_str("| Variant | ID switches/run | Fragments/run | New IDs/run | False assoc./run | Track coverage | Confirmed coverage | Same-ID reassoc. | Short-dropout survival | New tentative/run | Mean error | P95 error | Mean confirm s | Worst IDs / fragments |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
	for variant_name in [
		"A_no_prediction",
		"B_current_kalman",
		"C_weak",
		"C_medium",
		"C_strong",
	] {
		let summary = aggregate(
			results
				.iter()
				.filter(|result| result.variant.name() == variant_name),
		);
		report.push_str(&table_row(variant_name, &summary));
	}

	report.push_str("\n## A vs B by scenario\n\n");
	report.push_str("| Scenario | Variant | ID switches/run | Fragments/run | New IDs/run | False assoc./run | Track coverage | Confirmed coverage | Same-ID reassoc. | Short-dropout survival | New tentative/run | Mean error | P95 error | Mean confirm s | Worst IDs / fragments |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
	for scenario in Scenario::ALL {
		for variant in [Variant::NoPrediction, Variant::CurrentKalman] {
			let summary = aggregate(
				results
					.iter()
					.filter(|result| result.scenario == scenario && result.variant == variant),
			);
			let label = format!("{} | {}", scenario.name(), variant.name());
			report.push_str(&table_row(&label, &summary));
		}
	}

	report.push_str("\n## Motion-aware Q by PhoneMotion profile\n\n");
	report.push_str("| Profile / strength | ID switches/run | Fragments/run | New IDs/run | False assoc./run | Track coverage | Confirmed coverage | Same-ID reassoc. | Short-dropout survival | New tentative/run | Mean error | P95 error | Mean confirm s | Worst IDs / fragments |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
	for profile in PhoneProfile::ALL {
		for strength in [0.5, 1.0, 2.0] {
			let summary = aggregate(results.iter().filter(|result| {
				result.phone_profile == Some(profile)
					&& result.variant == Variant::MotionAware { strength }
			}));
			let label = format!(
				"{} / {}",
				profile.name(),
				Variant::MotionAware { strength }.name()
			);
			report.push_str(&table_row(&label, &summary));
		}
	}

	report.push_str("\n## Worst individual runs\n\n");
	report.push_str("| Scenario | Cadence | Seed | Variant | Phone profile | ID switches | Fragments | New IDs | False associations | Reassociation | Mean/P95 error |\n|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|\n");
	let mut worst = results.iter().collect::<Vec<_>>();
	worst.sort_by(|a, b| {
		let score = |result: &RunResult| {
			result.metrics.id_switches * 1000
				+ result.metrics.false_associations * 100
				+ result.metrics.fragments * 10
				+ result.metrics.new_ids
		};
		score(b).cmp(&score(a))
	});
	for result in worst.into_iter().take(20) {
		let m = &result.metrics;
		writeln!(
			report,
			"| {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {:.4}/{:.4} |",
			result.scenario.name(),
			result.cadence.name(),
			result.seed,
			result.variant.name(),
			result.phone_profile.map_or("none", PhoneProfile::name),
			m.id_switches,
			m.fragments,
			m.new_ids,
			m.false_associations,
			m.lost_reassociation_successes,
			m.lost_reassociation_opportunities,
			mean(&m.position_errors),
			percentile(&m.position_errors, 0.95),
		)
		.expect("writing to String cannot fail");
	}

	let mut cadence_map: BTreeMap<(&str, &str), Aggregate> = BTreeMap::new();
	for result in results
		.iter()
		.filter(|result| result.phone_profile.is_none())
	{
		cadence_map
			.entry((result.cadence.name(), result.variant.name()))
			.or_default()
			.add(&result.metrics);
	}
	report.push_str("\n## A vs B by cadence\n\n");
	report.push_str("| Cadence / variant | ID switches/run | Fragments/run | New IDs/run | False assoc./run | Track coverage | Confirmed coverage | Same-ID reassoc. | Short-dropout survival | New tentative/run | Mean error | P95 error | Mean confirm s | Worst IDs / fragments |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
	for ((cadence, variant), summary) in cadence_map {
		report.push_str(&table_row(&format!("{cadence} / {variant}"), &summary));
	}
	report
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	assert_production_baseline_equivalence();
	let results = run_simulation();
	let output_dir = PathBuf::from("target/kalman-motion-simulation");
	fs::create_dir_all(&output_dir)?;
	write_csv(&results, &output_dir)?;
	let summary = generated_summary(&results);
	fs::write(output_dir.join("summary.md"), &summary)?;
	println!("{summary}");
	println!(
		"Raw results: {}",
		output_dir.join("raw_results.csv").display()
	);
	Ok(())
}
