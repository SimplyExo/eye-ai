use std::{path::PathBuf, process::Command};

const REPO_URL: &str = "https://github.com/Peanutt42/ByteTrack-cpp";

const GIT_TAG: &str = "1.1.0";

fn main() {
	println!("cargo::rerun-if-changed=build.rs");

	let repo_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bytetrack-cpp");

	clone_repo(&repo_dir);

	match std::env::var("TARGET") {
		Ok(target) if target.contains("android") => {
			build_with_cmake_for_android(&repo_dir);
		}
		_ => build_with_cmake(&repo_dir),
	}

	link_with_cpp_stdlib();
}

fn clone_repo(repo_dir: &PathBuf) {
	let status = Command::new("git")
		.arg("clone")
		.arg("--branch")
		.arg(GIT_TAG)
		.arg("--depth")
		.arg("1")
		.arg(REPO_URL)
		.arg(repo_dir)
		.status()
		.expect("failed to run git clone to clone ByteTrack-cpp repo");

	if !status.success() {
		let status = Command::new("git")
			.arg("clean")
			.arg("-fdx")
			.current_dir(repo_dir)
			.status()
			.expect("failed to run git in order to clean local repo");

		assert!(status.success(), "failed to clone ByteTrack-cpp repo");
	}
}

fn build_with_cmake(repo_dir: &PathBuf) {
	let cmake_build_output = cmake::Config::new(repo_dir)
		.define("BUILD_BYTETRACK_TEST", "OFF")
		.no_build_target(true)
		.build_target("bytetrack")
		.build();

	println!(
		"cargo::rustc-link-search=native={}/build",
		cmake_build_output.display()
	);
	println!("cargo::rustc-link-lib=static=bytetrack");
}

fn build_with_cmake_for_android(repo_dir: &PathBuf) {
	println!("cargo::rerun-if-env-changed=ANDROID_NDK_ROOT");
	println!("cargo::rerun-if-env-changed=NDK_HOME");
	let ndk_dir = std::env::var("ANDROID_NDK_ROOT").or(std::env::var("NDK_HOME"))
		.expect("either set the environment variable `ANDROID_NDK_ROOT` or `NDK_HOME` to the ndk directory to build `bytetrack-cpp-rs` for android");
	let toolchain_file = PathBuf::from(&ndk_dir).join("build/cmake/android.toolchain.cmake");
	let target = std::env::var("TARGET").unwrap();
	let abi = match &*target {
		"aarch64-linux-android" => "arm64-v8a",
		"armv7-linux-androideabi" => "armeabi-v7a",
		"arm-linux-androideabi" => "armeabi",
		"thumbv7neon-linux-androideabi" => "armeabi", // TODO: is this correct?
		"i686-linux-android" => "x86",
		"x86_64-linux-android" => "x86_64",
		_ => unreachable!(),
	};
	let api_level = 21;
	let platform = format!("android-{}", api_level);

	let cmake_build_output = cmake::Config::new(repo_dir)
		.define("CMAKE_TOOLCHAIN_FILE", &toolchain_file)
		.define("ANDROID_ABI", abi)
		.define("ANDROID_NDK", &ndk_dir)
		.define("ANDROID_NATIVE_API_LEVEL", api_level.to_string())
		.define("ANDROID_PLATFORM", platform)
		.define("BUILD_BYTETRACK_TEST", "OFF")
		.no_build_target(true)
		.build_target("bytetrack")
		.build();

	println!(
		"cargo::rustc-link-search=native={}/build",
		cmake_build_output.display()
	);
	println!("cargo::rustc-link-lib=static=bytetrack");
}

fn link_with_cpp_stdlib() {
	let target = std::env::var("TARGET").unwrap();

	// TODO: not all cases tested!
	let cpp_stdlib = if target.contains("msvc") {
		None
	} else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
		Some("c++")
	} else if target.contains("android") {
		Some("c++_shared")
	} else {
		Some("stdc++")
	};

	if let Some(cpp_stdlib) = cpp_stdlib {
		println!("cargo::rustc-link-lib={}", cpp_stdlib);
	}
}
