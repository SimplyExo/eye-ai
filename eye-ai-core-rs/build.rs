use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
	println!("cargo::rerun-if-changed=build.rs");
	println!("cargo:rerun-if-env-changed=LITERT_LIB_DIR");
	println!("cargo:rerun-if-env-changed=LITERT_CACHE_DIR");

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

	// litert-sys caches prebuilt libLiteRt.so at:
	//   $XDG_CACHE_HOME/litert-sys/v0.10.2/<target>/
	// If we find it, add an rpath so the binary can find the .so at runtime.
	// We do this from the root crate because NixOS ld-wrapper strips
	// transitive rpath entries from upstream build scripts.
	if let Some(dir) = find_litert_cache_dir() {
		println!("cargo::rustc-link-arg=-Wl,-rpath,{}", dir.display());
	}
}

/// see https://docs.rs/crate/litert-sys/0.2.1/source/build.rs for more detail
fn find_litert_cache_dir() -> Option<PathBuf> {
	if let Ok(dir) = std::env::var("LITERT_LIB_DIR") {
		return Some(PathBuf::from(dir));
	}
	if let Ok(dir) = std::env::var("LITERT_CACHE_DIR") {
		return Some(PathBuf::from(dir));
	}

	let target = std::env::var("TARGET").ok()?;
	let dir = cache_dir_for(&target);

	if dir.join("libLiteRt.so").exists() {
		return Some(dir);
	}

	None
}

fn cache_root() -> PathBuf {
	if let Some(dir) = env::var_os("LITERT_CACHE_DIR") {
		return PathBuf::from(dir);
	}
	// Prefer the user-level cache. If we can't actually create it (e.g.,
	// running inside a container where `$HOME` points somewhere read-only,
	// as is the case in some cross-rs images), fall through to OUT_DIR so
	// the build doesn't panic before even attempting a download.
	if let Some(dir) = dirs::cache_dir()
		&& fs::create_dir_all(&dir).is_ok()
	{
		return dir;
	}
	PathBuf::from(env::var("OUT_DIR").unwrap()).join("litert-cache")
}

const LITERT_LM_TAG: &str = "v0.10.2";

fn cache_dir_for(target: &str) -> PathBuf {
	cache_root()
		.join("litert-sys")
		.join(LITERT_LM_TAG)
		.join(target)
}
