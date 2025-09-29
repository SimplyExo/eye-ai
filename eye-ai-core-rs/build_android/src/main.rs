use camino::Utf8Path;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};
use uniffi::KotlinBindingGenerator;
use uniffi_bindgen::EmptyCrateConfigSupplier;

fn main() {
	println!("Building eye-ai-core-rs-native-lib...");

	let native_lib_path = project_root().join("native_lib");

	let cargo_ndk_successfull = Command::new("cargo")
		.current_dir(native_lib_path)
		.args([
			"ndk",
			"-t",
			"arm64-v8a",
			"-o",
			"../../EyeAIApp/app/src/main/jniLibs",
			"build",
			"--release",
		])
		.status()
		.expect("failed to run cargo ndk")
		.success();

	if !cargo_ndk_successfull {
		eprintln!("failed to build eye-ai-core-rs-native-lib");
		exit(1);
	}

	println!("\nGenerating kotlin bindings for eye-ai-core-rs-native-lib...");
	generate_kotlin_bindings();
}

fn generate_kotlin_bindings() {
	let project_root = project_root();
	let library_path =
		project_root.join("target/aarch64-linux-android/release/libeye_ai_core_rs_native_lib.so");
	let out_dir =
		project_root.join("../EyeAIApp/app/src/main/java/com/algorithmic_alliance/eyeaiapp");

	uniffi::generate_bindings_library_mode::<KotlinBindingGenerator>(
		Utf8Path::new(&library_path.to_string_lossy().into_owned()),
		None,
		&KotlinBindingGenerator,
		&EmptyCrateConfigSupplier,
		None,
		Utf8Path::new(&out_dir.to_string_lossy().into_owned()),
		false,
	)
	.expect("Failed to generate kotlin bindings for android!");
}

fn project_root() -> PathBuf {
	Path::new(&env!("CARGO_MANIFEST_DIR"))
		.ancestors()
		.nth(1)
		.unwrap()
		.to_path_buf()
}
