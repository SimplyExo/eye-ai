use super::super::*;
use crate::{BoundingBox, DetectedObject, TrackedObject};
use std::{collections::HashMap, panic::AssertUnwindSafe};

fn depth_map() -> [f32; 256 * 256] {
	[2.0; 256 * 256]
}

fn labels() -> HashMap<String, ObjectLabelData> {
	HashMap::from([(
		"person".to_string(),
		ObjectLabelData {
			sample_begin: 0,
			sample_end: 1,
		},
	)])
}

fn tracked_object(center_x: f32, center_y: f32, tracking_id: i32) -> TrackedObject {
	TrackedObject::new(
		DetectedObject::new(
			"person".to_string(),
			tracking_id.max(0) as usize,
			0.9,
			BoundingBox::new(center_x, center_y, 0.2, 0.2),
		),
		tracking_id,
	)
}

fn process(objects: &[TrackedObject]) -> std::thread::Result<VecDeque<ObjectAudioSourceData>> {
	let map = depth_map();
	let label_data = labels();
	let profiling_frame = ProfilingFrame::new("bounds-test");
	std::panic::catch_unwind(AssertUnwindSafe(|| {
		process_object_detection_data(&map, objects, &label_data, &profiling_frame)
	}))
}

#[test]
fn normalized_coordinates_accept_only_finite_image_values() {
	for value in [0.0, 1.0] {
		assert!(normalized_depth_coordinate(value, 256).is_some());
	}
	for value in [
		-0.001,
		-1.0,
		1.001,
		2.0,
		f32::NAN,
		f32::INFINITY,
		f32::NEG_INFINITY,
	] {
		assert_eq!(
			normalized_depth_coordinate(value, 256),
			None,
			"value={value}"
		);
	}
}

#[test]
fn valid_center_and_all_image_edges_keep_existing_audio_mapping() {
	let cases = [(0.5, 0.5), (0.0, 0.5), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0)];

	for (index, (center_x, center_y)) in cases.into_iter().enumerate() {
		let output = process(&[tracked_object(center_x, center_y, index as i32)])
			.expect("valid coordinates must not panic");
		assert_eq!(output.len(), 1, "center=({center_x}, {center_y})");
		assert!(output[0].position.x.is_finite());
		assert!(output[0].position.y.is_finite());
	}
}

#[test]
fn invalid_coordinates_are_discarded_without_affecting_valid_objects() {
	let invalid = [
		(-0.01, 0.5),
		(0.5, -0.01),
		(1.01, 0.5),
		(0.5, 1.01),
		(f32::NAN, 0.5),
		(0.5, f32::NAN),
		(f32::INFINITY, 0.5),
		(0.5, f32::INFINITY),
		(f32::NEG_INFINITY, 0.5),
		(0.5, f32::NEG_INFINITY),
	];

	for (index, (center_x, center_y)) in invalid.into_iter().enumerate() {
		let output = process(&[tracked_object(center_x, center_y, index as i32)])
			.expect("invalid coordinates must not panic");
		assert!(output.is_empty(), "center=({center_x}, {center_y})");
	}

	let output = process(&[
		tracked_object(-0.1, 0.5, 1),
		tracked_object(0.5, 0.5, 2),
		tracked_object(f32::INFINITY, 0.5, 3),
		tracked_object(1.0, 1.0, 4),
	])
	.expect("mixed coordinates must not panic");
	assert_eq!(output.len(), 2);
	assert_eq!(
		output
			.iter()
			.map(|source| source.object_id)
			.collect::<Vec<_>>(),
		vec![2, 4],
	);
}
