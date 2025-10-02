use eye_ai_core_rs::{TensorFormat, TensorBuffer};

#[test]
fn test_tensor_buffer() {
	const FLOAT_FORMAT_A: TensorFormat = 1;
	const FLOAT_FORMAT_B: TensorFormat = 2;

	let mut container = [1.0f32, 2.0f32, 3.0f32];
	let tensor_buffer_a =
		TensorBuffer::<f32, FLOAT_FORMAT_A>::new(&mut container);

	for value in tensor_buffer_a.iter() {
		println!("{value}")
	}

	let tensor_buffer_b = tensor_buffer_a.convert_format::<FLOAT_FORMAT_B>();
	for value in tensor_buffer_b.iter() {
		println!("{value}")
	}
}
