use eye_ai_core_rs::{
	CreateDepthModelInfo, DepthModel, FLOAT_TENSOR_BUFFER_IMAGE_RGB_255_FORMAT,
	FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT, FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT,
	FloatTensorBuffer, ProfilingFrame,
};
use std::{
	ffi::{CStr, CString},
	path::PathBuf,
	sync::Arc,
};

#[test]
fn run_midas_depth_model() {
	tracing_tracy::client::Client::start();

	let input_image = image::ImageReader::open("tests/00022_00193_outdoor_010_030.png")
		.unwrap()
		.decode()
		.unwrap()
		.resize_exact(256, 256, image::imageops::FilterType::Nearest);
	let input_image_rgb8 = input_image.into_rgb8();
	let input_buffer_255f = input_image_rgb8
		.as_flat_samples()
		.as_slice()
		.iter()
		.map(|x| *x as f32)
		.collect::<Vec<_>>();
	let input =
		FloatTensorBuffer::<FLOAT_TENSOR_BUFFER_IMAGE_RGB_255_FORMAT>::new(input_buffer_255f);
	let input: FloatTensorBuffer<FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT> = input.into();

	let expected_output = ndarray_npy::read_npy::<_, ndarray::Array2<f32>>(
		"tests/00022_00193_outdoor_010_030_expected.npy",
	)
	.expect("failed to read expected output file");
	let flat_expected_output = expected_output.flatten();

	let tmp_dir = std::env::temp_dir().join("eye-ai-core-rs-testing-serialization-cache");
	std::fs::create_dir_all(&tmp_dir)
		.expect("failed to create tmp cache directory for delegate serialization");
	let tmp_dir_str = CString::new(tmp_dir.to_string_lossy().into_owned()).unwrap();

	let model_token = CString::new("midas_model_token").unwrap();

	let depth_profiling_frame = ProfilingFrame::new("Depth");

	let depth_model_create_info = CreateDepthModelInfo {
		tflite_lib_filepath: PathBuf::from("libtensorflow-lite.so"),
		tflite_gpu_delegate_lib_filepath: None,
		model_data: std::fs::read("../EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite")
			.expect("failed to load midas.tflite model file"),
		gpu_delegate_serialization_dir: tmp_dir_str,
		model_token,
		log_warning_callback: Arc::new(|message| println!("[WARN] {}", message)),
		log_error_callback: |message| unsafe {
			match CStr::from_ptr(message).to_str() {
				Ok(msg) => println!("[ERROR] {}", msg),
				Err(_) => println!(
					"[ERROR] Failed to convert CString to String in order to log tflite error message"
				),
			}
		},
		npu_config: None,
	};

	let mut depth_model = DepthModel::new(depth_model_create_info, &depth_profiling_frame)
		.expect("failed to create interpreter");

	let output_tensor_buffer: FloatTensorBuffer<FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT> =
		depth_model
			.run(input)
			.expect("failed to run inference on midas model");
	let output_tensor_buffer_data = output_tensor_buffer.data();

	let tolerance: f32 = 0.05;
	assert_eq!(output_tensor_buffer_data.len(), flat_expected_output.len());
	for i in 0..flat_expected_output.len() {
		let error = flat_expected_output[i] - output_tensor_buffer_data[i];
		assert!(
			error.abs() < tolerance,
			"error of {} at index {} (expected: {}, got: {})",
			error,
			i,
			flat_expected_output[i],
			output_tensor_buffer_data[i]
		);
	}
}
