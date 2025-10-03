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
	let third_party_dir = format!("third_party/{}", target_os);
	println!("cargo::rerun-if-changed={}", third_party_dir);
	let out_dir = std::env::var("OUT_DIR").unwrap();
	let target_output_dir = format!("{}/../../../", out_dir);

	for entry in std::fs::read_dir(&third_party_dir)
		.expect("failed to walk files of third_pary_dir")
		.flatten()
	{
		if entry.file_type().unwrap().is_file() {
			let path = entry.path();
			let filename = path.file_name().unwrap();
			std::fs::copy(
				entry.path(),
				format!("{}/{}", target_output_dir, filename.to_str().unwrap()),
			)
			.expect("failed to copy .so file to target output directory");
		}
	}
}
