fn main() {
	println!("cargo::rerun-if-changed=build.rs");
	println!("cargo::rerun-if-changed=src/tflite/tflite_error_reporter.cpp");
	cc::Build::new()
		.cpp(true)
		.file("src/tflite/tflite_error_reporter.cpp")
		.flag_if_supported("-std=c++20")
		.compile("eye_ai_core_rs_tflite_error_reporter");

	// copying .so files from third_party
	let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
	let out_dir = std::env::var("OUT_DIR").unwrap();
	let third_party_dir = format!("third_party/{}", target_os);

	// copy tensorflowlite .so file
	let libtensorflowlite_so_filename = if target_os == "android" {
		"libtensorflowlite_jni.so"
	} else {
		"libtensorflow-lite.so"
	};
	let libtensorflowlite_so = format!("{}/{}", third_party_dir, libtensorflowlite_so_filename);
	println!("cargo::rerun-if-changed={}", libtensorflowlite_so);
	std::fs::copy(
		libtensorflowlite_so,
		format!("{}/../../../{}", out_dir, libtensorflowlite_so_filename),
	)
	.expect("failed to copy tensorflow-lite library file");

	// copy libabsl_log_internal_nullguard.so
	let libabsl_log_internal_nullguard_so =
		format!("{}/libabsl_log_internal_nullguard.so", third_party_dir);
	println!(
		"cargo::rerun-if-changed={}",
		libabsl_log_internal_nullguard_so
	);
	std::fs::copy(
		libabsl_log_internal_nullguard_so,
		format!("{}/../../../libabsl_log_internal_nullguard.so", out_dir),
	)
	.expect("failed to copy absl log internal nullguard library file");

	// copy libqnn_delegate_jni.so
	if target_os == "android" {
		let qnn_delegate_library_so = format!("{}/libqnn_delegate_jni.so", third_party_dir);
		println!("cargo::rerun-if-changed={}", qnn_delegate_library_so);
		std::fs::copy(
			qnn_delegate_library_so,
			format!("{}/../../../libqnn_delegate_jni.so", out_dir),
		)
		.expect("failed to copy qnn delegate library file");
	}
}
