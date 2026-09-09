use std::ffi::CString;

use crate::tflite_runtime::sys;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpuConfig {
	pub config_type: sys::NpuConfigType,
	pub skel_library_dir: CString,
}
