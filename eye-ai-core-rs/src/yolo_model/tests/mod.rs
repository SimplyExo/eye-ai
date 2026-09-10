use super::*;

fn object(
	center_x: f32,
	center_y: f32,
	width: f32,
	height: f32,
	confidence: f32,
) -> DetectedObject {
	DetectedObject::new(
		format!("object{confidence}"),
		0,
		confidence,
		BoundingBox::new(center_x, center_y, width, height),
	)
}

fn assert_close(actual: f32, expected: f32) {
	assert!(
		(actual - expected).abs() < 1e-6,
		"expected {expected}, got {actual}"
	);
}

#[test]
fn identical_boxes_have_iou_one() {
	let a = object(0.5, 0.5, 0.4, 0.4, 0.9);
	assert_close(calculate_iou(&a, &a), 1.0);
}

#[test]
fn separated_boxes_have_iou_zero() {
	let a = object(0.2, 0.5, 0.2, 0.2, 0.9);
	let b = object(0.8, 0.5, 0.2, 0.2, 0.8);
	assert_close(calculate_iou(&a, &b), 0.0);
}

#[test]
fn partial_overlap_has_analytical_iou() {
	let a = object(0.2, 0.2, 0.4, 0.4, 0.9);
	let b = object(0.4, 0.2, 0.4, 0.4, 0.8);
	assert_close(calculate_iou(&a, &b), 1.0 / 3.0);
}

#[test]
fn degenerate_boxes_return_zero_instead_of_nan() {
	let a = object(0.5, 0.5, 0.0, 0.0, 0.9);
	let b = object(0.5, 0.5, 0.0, 0.0, 0.8);
	assert_close(calculate_iou(&a, &b), 0.0);
}

#[test]
fn nms_keeps_disjoint_boxes() {
	let frame = ProfilingFrame::new("nms_iou_tests");
	let low = object(0.2, 0.2, 0.2, 0.2, 0.6);
	let high = object(0.8, 0.2, 0.2, 0.2, 0.9);
	let selected = apply_nms(&[low, high], 0.5, &frame);
	assert_eq!(selected.len(), 2);
	assert_close(selected[0].confidence, 0.9);
	assert_close(selected[1].confidence, 0.6);
}

#[test]
fn nms_keeps_boxes_below_the_iou_threshold() {
	let frame = ProfilingFrame::new("nms_iou_tests");
	let low = object(0.2, 0.2, 0.4, 0.4, 0.6);
	let high = object(0.4, 0.2, 0.4, 0.4, 0.9);
	let selected = apply_nms(&[low, high], 0.5, &frame);
	assert_eq!(selected.len(), 2);
}

#[test]
fn nms_suppresses_sufficiently_overlapping_boxes() {
	let frame = ProfilingFrame::new("nms_iou_tests");
	let low = object(0.2, 0.2, 0.4, 0.4, 0.6);
	let high = object(0.25, 0.2, 0.4, 0.4, 0.9);
	let selected = apply_nms(&[low, high], 0.5, &frame);
	assert_eq!(selected.len(), 1);
	assert_close(selected[0].confidence, 0.9);
}
