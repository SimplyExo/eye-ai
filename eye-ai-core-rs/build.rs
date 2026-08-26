fn main() {
	println!("cargo::rerun-if-changed=build.rs");

	// build tflite-runtime c++ cmake library
	let tflite_runtime_dst = cmake::Config::new("../tflite-runtime/")
		.define("CMAKE_BUILD_TYPE", "Release")
		// fixes some older deps of tensorflow-lite
		.define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
		.build();

	// Read and echo the link manifest
	let manifest_path = tflite_runtime_dst.join("build/tflite-cargo-link.txt");
	println!("cargo::rerun-if-changed={}", manifest_path.display());
	let manifest = std::fs::read_to_string(&manifest_path)
		.expect("failed to read tflite-cargo-link.txt, did the CMake build succeed?");
	for line in manifest.lines() {
		if !line.is_empty() && !line.starts_with('#') {
			println!("{}", line);
		}
	}

	println!("cargo:rerun-if-changed=../tflite-runtime/CMakeLists.txt");
	println!("cargo:rerun-if-changed=../tflite-runtime/include");
	println!("cargo:rerun-if-changed=../tflite-runtime/src");
	println!("cargo:rerun-if-changed=../tflite-runtime/third_party");
}
