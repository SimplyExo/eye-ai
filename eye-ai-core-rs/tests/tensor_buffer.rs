use eye_ai_core_rs::{FloatTensorBuffer, TensorFormat};

fn assert_slice_equal(a: &[f32], b: &[f32]) {
	assert_eq!(a.len(), b.len());
	for i in 0..a.len() {
		assert_eq!(a[i], b[i]);
	}
}

#[test]
fn test_tensor_buffer() {
	const FLOAT_FORMAT_A: TensorFormat = 1;
	const FLOAT_FORMAT_B: TensorFormat = 2;

	let mut container = [1.0f32, 2.0f32, 3.0f32];
	let mut container_copy = container;
	let tensor_buffer_a = FloatTensorBuffer::<FLOAT_FORMAT_A>::new(&mut container);

	assert_slice_equal(&container_copy, tensor_buffer_a.data());

	let mut tensor_buffer_b = tensor_buffer_a.convert_format::<FLOAT_FORMAT_B>();
	tensor_buffer_b.data_mut()[2] = 4.0f32;
	container_copy[2] = 4.0f32;
	assert_slice_equal(&container_copy, tensor_buffer_b.data());
}
