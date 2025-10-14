use bytetrack_cpp_rs::{BYTETracker, Object, Rect};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

const EPS: f64 = 1e-2;
const D_RESULTS_FILE: &str = "tests/detection_results.json";
const T_RESULTS_FILE: &str = "tests/tracking_results.json";

#[derive(Deserialize)]
struct TrackingResult {
	frame_id: String,
	track_id: String,
	x: String,
	y: String,
	width: String,
	height: String,
}

#[derive(Deserialize)]
struct DetectionResult {
	frame_id: String,
	prob: String,
	x: String,
	y: String,
	width: String,
	height: String,
}

#[derive(Deserialize)]
struct TestingResults<T> {
	name: String,
	fps: i32,
	track_buffer: i32,
	results: Vec<T>,
}

type BYTETrackerOut = HashMap<usize, Rect>;

fn get_inputs_ref(
	pt: &TestingResults<DetectionResult>,
) -> Result<HashMap<usize, Vec<Object>>, Box<dyn std::error::Error>> {
	let mut inputs_ref = HashMap::new();

	for result in pt.results.iter() {
		let frame_id: usize = result.frame_id.parse()?;
		let prob: f32 = result.prob.parse()?;
		let x: f32 = result.x.parse()?;
		let y: f32 = result.y.parse()?;
		let width: f32 = result.width.parse()?;
		let height: f32 = result.height.parse()?;

		let rect = Rect::new(x, y, width, height);
		let object = Object::new(rect, 0, prob);

		inputs_ref
			.entry(frame_id)
			.or_insert_with(Vec::new)
			.push(object);
	}

	Ok(inputs_ref)
}

fn get_outputs_ref(
	pt: &TestingResults<TrackingResult>,
) -> Result<HashMap<usize, BYTETrackerOut>, Box<dyn std::error::Error>> {
	let mut outputs_ref = HashMap::new();

	for result in pt.results.iter() {
		let frame_id: usize = result.frame_id.parse()?;
		let track_id: usize = result.track_id.parse()?;
		let x: f32 = result.x.parse()?;
		let y: f32 = result.y.parse()?;
		let width: f32 = result.width.parse()?;
		let height: f32 = result.height.parse()?;

		let rect = Rect::new(x, y, width, height);

		outputs_ref
			.entry(frame_id)
			.or_insert_with(HashMap::new)
			.insert(track_id, rect);
	}

	Ok(outputs_ref)
}

#[test]
fn bytetracker_basic() {
	let d_results_file = File::open(D_RESULTS_FILE).expect("Failed to open detection results file");
	let pt_d_results: TestingResults<DetectionResult> =
		serde_json::from_reader(d_results_file).expect("Failed to parse detection results JSON");

	let t_results_file = File::open(T_RESULTS_FILE).expect("Failed to open tracking results file");
	let pt_t_results: TestingResults<TrackingResult> =
		serde_json::from_reader(t_results_file).expect("Failed to parse tracking results JSON");

	let result = || -> Result<(), Box<dyn std::error::Error>> {
		let detection_results_name: &String = &pt_d_results.name;
		let tracking_results_name: &String = &pt_t_results.name;
		let fps: i32 = pt_d_results.fps;
		let track_buffer: i32 = pt_d_results.track_buffer;

		if detection_results_name != tracking_results_name {
			return Err(format!(
				"The name of the tests are different: [detection_results_name: {}, tracking_results_name: {}]",
				detection_results_name, tracking_results_name
			)
			.into());
		}

		let inputs_ref = get_inputs_ref(&pt_d_results)?;

		let outputs_ref = get_outputs_ref(&pt_t_results)?;

		// prev: (frames) max_time_lost = (framerate / 30) * track_buffer
		// now: (frames) max_time_lost = framerate * max_time_lost_seconds
		let max_time_lost_seconds = track_buffer as f32 / 30.0;

		let mut tracker = BYTETracker::new(
			max_time_lost_seconds,
			fps as f32,
			BYTETracker::DEFAULT_TRACK_THRESHOLD,
			BYTETracker::DEFAULT_HIGH_THRESHOLD,
			BYTETracker::DEFAULT_MATCH_THRESHOLD,
		);

		let mut frames: Vec<&usize> = inputs_ref.keys().collect();
		frames.sort();

		for &frame_id in frames {
			let objects = &inputs_ref[&frame_id];
			let outputs = tracker.update(objects);

			let ref_outputs = &outputs_ref[&frame_id];
			assert_eq!(
				outputs.len(),
				ref_outputs.len(),
				"Frame {}: Output count mismatch",
				frame_id
			);

			for output in outputs {
				let rect = &output.rect;
				let track_id = output.track_id as usize;

				if let Some(ref_rect) = ref_outputs.get(&track_id) {
					assert!(
						(ref_rect.x - rect.x).abs() < EPS as f32,
						"Frame {} Track {}: x coordinate mismatch - expected: {}, got: {}",
						frame_id,
						track_id,
						ref_rect.x,
						rect.x
					);
					assert!(
						(ref_rect.y - rect.y).abs() < EPS as f32,
						"Frame {} Track {}: y coordinate mismatch - expected: {}, got: {}",
						frame_id,
						track_id,
						ref_rect.y,
						rect.y
					);
					assert!(
						(ref_rect.width - rect.width).abs() < EPS as f32,
						"Frame {} Track {}: width mismatch - expected: {}, got: {}",
						frame_id,
						track_id,
						ref_rect.width,
						rect.width
					);
					assert!(
						(ref_rect.height - rect.height).abs() < EPS as f32,
						"Frame {} Track {}: height mismatch - expected: {}, got: {}",
						frame_id,
						track_id,
						ref_rect.height,
						rect.height
					);
				} else {
					panic!(
						"Frame {}: Track ID {} not found in reference data",
						frame_id, track_id
					);
				}
			}
		}

		Ok(())
	}();

	if let Err(e) = result {
		panic!("Test failed: {}", e);
	}
}
