use crate::{sys, Delegate};
use libloading::Library;
use std::ffi::{CString, OsStr};
use std::sync::Arc;
use thiserror::Error;

macro_rules! load_gpu_function {
	($library:ident, $function:ident) => {{
		let function_name_with_f = stringify!($function);
		let function_name = &function_name_with_f[..function_name_with_f.len() - 1];
		let symbol_result = unsafe {
			$library.get::<$function>(CString::new(function_name).unwrap().as_bytes_with_nul())
		};
		if let Ok(symbol) = symbol_result {
			*symbol
		} else {
			return Err(LoadTfLiteGpuDelegateError::MissingFunction(
				function_name.to_string(),
			));
		}
	}};
}

#[derive(Debug, Error)]
pub enum LoadTfLiteGpuDelegateError {
	#[error("Failed to load gpu delegate library: {0}")]
	Loading(#[from] libloading::Error),
	#[error("Missing function '{0}' in gpu delegate library")]
	MissingFunction(String),
}

pub struct GpuDelegateAPI {
	_vt: Arc<sys::TfLiteVt>,
	// needed for lifetime reasons
	#[allow(unused)]
	library: Arc<Library>,
	delegate_options_v2_default: TfLiteGpuDelegateOptionsV2DefaultF,
	delegate_v2_create: TfLiteGpuDelegateV2CreateF,
	delegate_v2_delete: TfLiteGpuDelegateV2DeleteF,
}
impl GpuDelegateAPI {
	pub fn load_from_separate_library<P: AsRef<OsStr>>(
		_vt: Arc<sys::TfLiteVt>,
		path: P,
	) -> Result<Self, LoadTfLiteGpuDelegateError> {
		let library =
			Arc::new(unsafe { Library::new(path).map_err(LoadTfLiteGpuDelegateError::Loading) }?);

		Self::load(_vt, library)
	}

	pub fn load(
		_vt: Arc<sys::TfLiteVt>,
		library: Arc<Library>,
	) -> Result<Self, LoadTfLiteGpuDelegateError> {
		let delegate_options_v2_default =
			load_gpu_function!(library, TfLiteGpuDelegateOptionsV2DefaultF);
		let delegate_v2_create = load_gpu_function!(library, TfLiteGpuDelegateV2CreateF);
		let delegate_v2_delete = load_gpu_function!(library, TfLiteGpuDelegateV2DeleteF);

		Ok(Self {
			_vt,
			library,
			delegate_options_v2_default,
			delegate_v2_create,
			delegate_v2_delete,
		})
	}

	pub fn default_options(&self) -> TfLiteGpuDelegateOptionsV2 {
		unsafe { (self.delegate_options_v2_default)() }
	}

	pub fn create_delegate(&self, options: &TfLiteGpuDelegateOptionsV2) -> Delegate {
		let delegate = unsafe { (self.delegate_v2_create)(options) };

		Delegate::new(self._vt.clone(), delegate, self.delegate_v2_delete)
	}
}

pub type TfLiteGpuDelegateOptionsV2DefaultF = unsafe extern "C" fn() -> TfLiteGpuDelegateOptionsV2;

pub type TfLiteGpuDelegateV2CreateF =
	unsafe extern "C" fn(options: *const TfLiteGpuDelegateOptionsV2) -> *mut sys::TfLiteDelegate;

pub type TfLiteGpuDelegateV2DeleteF = unsafe extern "C" fn(delegate: *mut sys::TfLiteDelegate);

#[repr(i32)]
pub enum TfLiteGpuInferenceUsage {
	FastSingleAnswer = 0,
	SustainedSpeed = 1,
	Balanced = 2,
}

#[repr(i32)]
pub enum TfLiteGpuInferencePriority {
	Auto = 0,
	MaxPrecision = 1,
	MinLatency = 2,
	MinMemoryUsage = 3,
}

#[repr(i64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TfLiteGpuExperimentalFlags {
	None = 0,
	EnableQuant = 1 << 0,
	CLOnly = 1 << 1,
	GLOnly = 1 << 2,
	EnableSerialization = 1 << 3,
}

#[repr(C)]
pub struct TfLiteGpuDelegateOptionsV2 {
	pub is_precision_loss_allowed: i32,
	pub inference_preference: TfLiteGpuInferenceUsage,
	pub inference_priority1: TfLiteGpuInferencePriority,
	pub inference_priority2: TfLiteGpuInferencePriority,
	pub inference_priority3: TfLiteGpuInferencePriority,
	/// use TfLiteGpuExperimentalFlags
	pub experimental_flags: i64,
	pub max_delegated_partitions: i32,
	pub serialization_dir: *const std::os::raw::c_char,
	pub model_token: *const std::os::raw::c_char,
}
