use crate::{Rect, sys::byte_track_Rect_float_calc_iou_for_testing};

const EPSILON: f32 = 1e-5;

fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
	Rect::new(x, y, width, height)
}

fn native_iou(first: Rect, second: Rect) -> f32 {
	unsafe { byte_track_Rect_float_calc_iou_for_testing(first, second) }
}

fn assert_iou(actual: f32, expected: f32) {
	assert!(actual.is_finite(), "IoU must be finite, got {actual}");
	assert!(
		(actual - expected).abs() <= EPSILON,
		"expected {expected}, got {actual}"
	);
	assert!((0.0..=1.0).contains(&actual), "IoU must be in [0, 1]");
}

#[test]
fn identical_boxes_have_iou_one() {
	let box_ = rect(0.10, 0.20, 0.30, 0.40);
	assert_iou(native_iou(box_, box_), 1.0);
}

#[test]
fn separated_normalized_boxes_have_iou_zero() {
	assert_iou(
		native_iou(rect(0.00, 0.00, 0.10, 0.10), rect(0.80, 0.80, 0.10, 0.10)),
		0.0,
	);
}

#[test]
fn partial_overlap_uses_continuous_area() {
	// Intersection = 0.10 * 0.20, union = 0.20 * 0.20 * 2 - intersection.
	assert_iou(
		native_iou(rect(0.10, 0.10, 0.20, 0.20), rect(0.20, 0.10, 0.20, 0.20)),
		1.0 / 3.0,
	);
}

#[test]
fn contained_box_uses_the_larger_box_as_union() {
	// Intersection = 0.20 * 0.20, union = 0.60 * 0.60.
	assert_iou(
		native_iou(rect(0.10, 0.10, 0.60, 0.60), rect(0.20, 0.20, 0.20, 0.20)),
		1.0 / 9.0,
	);
}

#[test]
fn tiny_and_typical_yolo_boxes_have_analytical_iou() {
	// Tiny normalized boxes: intersection = 0.005 * 0.01, union = 0.00015.
	assert_iou(
		native_iou(rect(0.10, 0.10, 0.01, 0.01), rect(0.105, 0.10, 0.01, 0.01)),
		1.0 / 3.0,
	);
	// Typical YOLO boxes: intersection = 0.15 * 0.40, union = 0.19.
	assert_iou(
		native_iou(rect(0.10, 0.10, 0.25, 0.50), rect(0.20, 0.20, 0.25, 0.50)),
		6.0 / 19.0,
	);
}

#[test]
fn image_edges_and_degenerate_boxes_are_safe() {
	assert_iou(
		native_iou(rect(0.00, 0.00, 0.20, 0.20), rect(0.80, 0.80, 0.20, 0.20)),
		0.0,
	);
	assert_iou(
		native_iou(rect(0.80, 0.80, 0.20, 0.20), rect(0.80, 0.80, 0.20, 0.20)),
		1.0,
	);

	for invalid in [
		rect(0.10, 0.10, 0.00, 0.20),
		rect(0.10, 0.10, 0.20, 0.00),
		rect(0.10, 0.10, -0.20, 0.20),
		rect(f32::NAN, 0.10, 0.20, 0.20),
		rect(0.10, f32::INFINITY, 0.20, 0.20),
	] {
		assert_iou(native_iou(invalid, rect(0.10, 0.10, 0.20, 0.20)), 0.0);
	}
}
