// C++ dependencies reference ARM64 outlined atomics absent from Rust builtins.
fn main() {
	println!("cargo:rerun-if-env-changed=NDK_HOME");
	println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");
	if std::env::var("TARGET").as_deref() != Ok("aarch64-linux-android") {
		return;
	}
	let ndk = std::env::var("NDK_HOME")
		.or_else(|_| std::env::var("ANDROID_NDK_ROOT"))
		.expect("Android build requires NDK_HOME");
	let prebuilts = std::path::Path::new(&ndk).join("toolchains/llvm/prebuilt");
	for host in std::fs::read_dir(prebuilts).expect("NDK prebuilts") {
		let clang = host.expect("NDK host").path().join("lib/clang");
		if let Ok(versions) = std::fs::read_dir(clang) {
			for version in versions.flatten() {
				let archive = version
					.path()
					.join("lib/linux/libclang_rt.builtins-aarch64-android.a");
				if archive.is_file() {
					println!("cargo:rustc-link-arg={}", archive.display());
					return;
				}
			}
		}
	}
	panic!("NDK ARM64 compiler runtime archive missing");
}
