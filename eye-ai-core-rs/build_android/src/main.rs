use camino::Utf8Path;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};
use uniffi::KotlinBindingGenerator;
use uniffi_bindgen::EmptyCrateConfigSupplier;

fn main() {
	let additional_args: Vec<String> = std::env::args().skip(1).collect();

	if !additional_args.is_empty() {
		println!("Additional args: {}", additional_args.join(" "));
	}

	println!("Building eye-ai-core-rs-native-lib...");

	// root dir of eye-ai-core-rs, not the entire eye-ai git repo
	let project_root = project_root();

	let native_lib_path = project_root.join("native_lib");

	let arch = "arm64-v8a";
	let eye_ai_app_output_dir = project_root.join("../EyeAIApp/app/src/main/jniLibs");
	let eye_ai_app_libraries_dir = eye_ai_app_output_dir.join(arch);

	let ndk_home_set = std::env::var("NDK_HOME").is_ok();

	let mut cargo_ndk_command = Command::new("cargo");

	cargo_ndk_command
		.current_dir(native_lib_path)
		.args([
			"ndk",
			"-t",
			arch,
			"-o",
			eye_ai_app_output_dir.to_str().unwrap(),
			"build",
			"--release",
		])
		.args(additional_args);

	if !ndk_home_set {
		let ndk_path = std::env::var("ANDROID_NDK_ROOT").expect(
			"neither 'NDK_HOME' nor 'ANDROID_NDK_ROOT' environment variable is set, please set one of them",
		);
		cargo_ndk_command.env("NDK_HOME", ndk_path);
	}

	// cargo-ndk's cc_env() falls back to the plain CC/CXX variable when
	// it is already set in the environment, and then overwrites it with
	// the NDK path.  This pollutes the build for host-target build-deps
	// (e.g. ring, bzip2-sys) which then try to compile with the NDK
	// clang and fail on missing host headers like <assert.h> and <stdlib.h>.
	// Strip CC/CXX so cargo-ndk uses target-specific variables instead
	// (e.g. CC_aarch64-linux-android) which only affect the Android target.
	cargo_ndk_command.env_remove("CC");
	cargo_ndk_command.env_remove("CXX");

	let cargo_ndk_status = cargo_ndk_command.status().expect("failed to run cargo ndk");

	if !cargo_ndk_status.success() {
		eprintln!("failed to build eye-ai-core-rs-native-lib");
		exit(cargo_ndk_status.code().unwrap_or(1));
	}

	println!("\nCopying third party libraries from eye-ai-core-rs to EyeAIApp...");
	let third_party_dir = project_root.join("third_party/android");
	for file in std::fs::read_dir(third_party_dir)
		.expect("failed to iterate over third party libraries for android")
		.flatten()
	{
		let src_path = file.path();
		if src_path.is_file() {
			let dst_path =
				eye_ai_app_libraries_dir.join(src_path.file_name().expect("expected file name"));

			std::fs::copy(src_path, dst_path)
				.expect("failed to copy third party library from eye-ai-core-rs to EyeAIApp");
		}
	}

	println!("\nCopying third party libraries from tflite-runtime to EyeAIApp...");
	let tflite_runtime_third_party_dir = project_root.join("../tflite-runtime/third_party/");
	for dependency in std::fs::read_dir(tflite_runtime_third_party_dir)
		.expect("failed to iterate over third party libraries for android")
		.flatten()
	{
		let dependency_path = dependency.path();
		if dependency_path.is_dir() {
			for src in std::fs::read_dir(dependency_path.join("lib/arm64-v8a"))
				.expect("failed to iterate over tflite dependencies arm64-v8a libraries")
				.flatten()
			{
				let src_path = src.path();
				let dst_path = eye_ai_app_libraries_dir
					.join(src_path.file_name().expect("expected file name"));

				std::fs::copy(src_path, dst_path)
					.expect("failed to copy third party library from tflite-runtime to EyeAIApp");
			}
		}
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
