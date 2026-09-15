extern crate cmake;

use cmake::Config;
use std::{env, path::PathBuf, process::Command};

// can be overridden by setting the env var ANDROID_NATIVE_API_LEVEL
const DEFAULT_ANDROID_API_LEVEL: &str = "21";

const OPENAL_SOFT_TAG: &str = "1.24.3";

const OPENAL_REPO: &str = "https://github.com/kcat/openal-soft.git";

fn clone_openalsoft() -> PathBuf {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("openal-soft");
    let status = Command::new("git")
        .arg("clone")
        .args(["--branch", OPENAL_SOFT_TAG])
        .args(["--depth", "1"])
        .arg(OPENAL_REPO)
        .arg(&out)
        .status()
        .unwrap();
    if !status.success() {
        let status = Command::new("git")
            .arg("clean")
            .arg("-fdx")
            .current_dir(&out)
            .status()
            .unwrap();
        assert!(status.success(), "failed to clone openal-soft");
        let status = Command::new("git")
            .arg("checkout")
            .arg(format!("tags/{}", OPENAL_SOFT_TAG))
            .current_dir(&out)
            .status()
            .unwrap();
        assert!(status.success(), "failed to clone openal-soft");
    }
    out
}

#[cfg(not(feature = "dynamic"))]
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

fn build_openalsoft(openal_dir: PathBuf) {
    let libtype = match env::var("CARGO_FEATURE_DYNAMIC") {
        Ok(_) => "SHARED",
        _ => "STATIC",
    };
    let dst = Config::new(openal_dir)
        .define("ALSOFT_UTILS", "OFF")
        .define("ALSOFT_EXAMPLES", "OFF")
        .define("ALSOFT_TESTS", "OFF")
        .define("LIBTYPE", libtype)
        .no_build_target(true)
        .build();
    println!("cargo:rustc-link-search=native={}/build", dst.display());

    let link_type = match env::var("CARGO_FEATURE_DYNAMIC") {
        Ok(_) => "dylib",
        _ => "static",
    };
    // openal-soft does not build this anymore?! println!("cargo:rustc-link-lib={}=common", link_type);
    println!("cargo:rustc-link-lib={}=openal", link_type);
}

fn build_openalsoft_android(openal_dir: PathBuf) {
    let target = &*env::var("TARGET").unwrap();
    let ndk_dir = &*env::var("NDK_HOME").expect("set the environment variable `NDK_HOME` to the ndk directory to build `al-sys` for android");

    let toolchain_file = PathBuf::from(ndk_dir).join("build/cmake/android.toolchain.cmake");
    let abi = match target {
        "aarch64-linux-android" => "arm64-v8a",
        "armv7-linux-androideabi" => "armeabi-v7a",
        "arm-linux-androideabi" => "armeabi",
        "thumbv7neon-linux-androideabi" => "armeabi", // TODO: is this correct?
        "i686-linux-android" => "x86",
        "x86_64-linux-android" => "x86_64",
        _ => unreachable!(),
    };
    let libtype = match env::var("CARGO_FEATURE_DYNAMIC") {
        Ok(_) => "SHARED",
        _ => "STATIC",
    };
    let api_level =
        env::var("ANDROID_NATIVE_API_LEVEL").unwrap_or(DEFAULT_ANDROID_API_LEVEL.to_owned());
    let platform = &*format!("android-{}", api_level);

    let dst = Config::new(openal_dir)
        .define("CMAKE_TOOLCHAIN_FILE", toolchain_file)
        .define("ANDROID_ABI", abi)
        .define("ALSOFT_UTILS", "OFF")
        .define("ALSOFT_EXAMPLES", "OFF")
        .define("ALSOFT_TESTS", "OFF")
        .define("ANDROID_NDK", ndk_dir)
        .define("LIBTYPE", libtype)
        .define("ANDROID_NATIVE_API_LEVEL", api_level)
        .define("ANDROID_PLATFORM", platform)
        .no_build_target(true)
        .build();
    println!("cargo:rerun-if-env-changed=ANDROID_NATIVE_API_LEVEL");
    println!("cargo:rerun-if-env-changed=NDK_HOME");
    println!("cargo:rustc-link-search=native={}/build", dst.display());

    let link_type = match env::var("CARGO_FEATURE_DYNAMIC") {
        Ok(_) => "dylib",
        _ => "static",
    };
    // openal-soft does not build this anymore?! println!("cargo:rustc-link-lib={}=common", link_type);
    println!("cargo:rustc-link-lib={}=openal", link_type);
}

fn main() {
    let target = &*env::var("TARGET").unwrap();
    match target {
        "aarch64-linux-android"
        | "armv7-linux-androideabi"
        | "arm-linux-androideabi"
        | "thumbv7neon-linux-androideabi"
        | "i686-linux-android"
        | "x86_64-linux-android" => {
            let repo_path = clone_openalsoft();
            build_openalsoft_android(repo_path);
        }
        _ => {
            let repo_path = clone_openalsoft();
            build_openalsoft(repo_path)
        }
    }

    #[cfg(not(feature = "dynamic"))]
    link_with_cpp_stdlib();
}
