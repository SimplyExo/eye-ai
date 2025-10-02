use std::ffi::{CString, OsStr};
use std::fmt::Display;
use std::sync::Arc;

use libloading::Library;
use thiserror::Error;

use crate::sys::{self, TfLiteVt};
use crate::Delegate;

macro_rules! load_qnn_function {
	($library:ident, $function:ident) => {{
		let function_name_with_f = stringify!($function);
		let mut function_name = String::from(function_name_with_f);
		function_name.truncate(1);
		let symbol_result = unsafe {
			$library.get::<$function>(
				CString::new(function_name.clone())
					.unwrap()
					.as_bytes_with_nul(),
			)
		};
		if let Ok(symbol) = symbol_result {
			*symbol
		} else {
			return Err(LoadQnnDelegateLibError::MissingFunction(function_name));
		}
	}};
}

#[derive(Debug, Error)]
pub enum LoadQnnDelegateLibError {
	#[error("Failed to load QNN delegate library: {0}")]
	Load(#[from] libloading::Error),
	#[error("Missing function '{0}' in QNN delegate library")]
	MissingFunction(String),
	#[error("Version mismatch: library has version {library_version} but this code was compiled for {expected_version}")]
	VersionMismatch {
		library_version: QnnDelegateApiVersion,
		expected_version: QnnDelegateApiVersion,
	},
}

pub struct QnnDelegateVt {
	// needed for lifetime reasons
	#[allow(unused)]
	library: Arc<Library>,
	_vt: Arc<TfLiteVt>,
	options_default: TfLiteQnnDelegateOptionsDefaultF,
	delegate_create: TfLiteQnnDelegateCreateF,
	delegate_delete: TfLiteQnnDelegateDeleteF,
	/*delegate_set_perf: TfLiteQnnDelegateSetPerfF,
	delegate_has_capability: TfLiteQnnDelegateHasCapabilityF,
	delegate_update_htp_perf_mode: TfLiteQnnDelegateUpdateHtpPerfModeF,
	delegate_update_dsp_perf_mode: TfLiteQnnDelegateUpdateDspPerfModeF,
	delegate_get_api_version: TfLiteQnnDelegateGetApiVersionF,
	delegate_alloc_custom_mem: TfLiteQnnDelegateAllocCustomMemF,
	delegate_free_custom_mem: TfLiteQnnDelegateFreeCustomMemF,
	delegate_get_profiling_result: TfLiteQnnDelegateGetProfilingResultF,
	delegate_clear_profiling_result: TfLiteQnnDelegateClearProfilingResultF,*/
}
impl QnnDelegateVt {
	pub fn load<P: AsRef<OsStr>>(
		_vt: Arc<TfLiteVt>,
		path: P,
	) -> Result<Self, LoadQnnDelegateLibError> {
		let library = unsafe { Library::new(path) }?;
		let options_default = load_qnn_function!(library, TfLiteQnnDelegateOptionsDefaultF);
		let delegate_create = load_qnn_function!(library, TfLiteQnnDelegateCreateF);
		let delegate_delete = load_qnn_function!(library, TfLiteQnnDelegateDeleteF);
		let delegate_get_api_version = load_qnn_function!(library, TfLiteQnnDelegateGetApiVersionF);
		/*let delegate_set_perf = load_qnn_function!(library, TfLiteQnnDelegateSetPerfF);
		let delegate_has_capability = load_qnn_function!(library, TfLiteQnnDelegateHasCapabilityF);
		let delegate_update_htp_perf_mode =
			load_qnn_function!(library, TfLiteQnnDelegateUpdateHtpPerfModeF);
		let delegate_update_dsp_perf_mode =
			load_qnn_function!(library, TfLiteQnnDelegateUpdateDspPerfModeF);
		let delegate_alloc_custom_mem =
			load_qnn_function!(library, TfLiteQnnDelegateAllocCustomMemF);
		let delegate_free_custom_mem = load_qnn_function!(library, TfLiteQnnDelegateFreeCustomMemF);
		let delegate_get_profiling_result =
			load_qnn_function!(library, TfLiteQnnDelegateGetProfilingResultF);
		let delegate_clear_profiling_result =
			load_qnn_function!(library, TfLiteQnnDelegateClearProfilingResultF);*/

		let version = unsafe { (delegate_get_api_version)() };
		if version != RS_PORT_QNN_DELEGATE_API_VERSION {
			return Err(LoadQnnDelegateLibError::VersionMismatch {
				library_version: version,
				expected_version: RS_PORT_QNN_DELEGATE_API_VERSION,
			});
		}

		Ok(Self {
			library: Arc::new(library),
			_vt,
			options_default,
			delegate_create,
			delegate_delete,
			/*delegate_set_perf,
			delegate_has_capability,
			delegate_update_htp_perf_mode,
			delegate_update_dsp_perf_mode,
			delegate_get_api_version,
			delegate_alloc_custom_mem,
			delegate_free_custom_mem,
			delegate_get_profiling_result,
			delegate_clear_profiling_result,*/
		})
	}

	pub fn default_options(&self) -> TfLiteQnnDelegateOptions {
		unsafe { (self.options_default)() }
	}

	pub fn create_delegate(&self, options: &TfLiteQnnDelegateOptions) -> Delegate {
		let delegate = unsafe { (self.delegate_create)(options) };
		Delegate::new(self._vt.clone(), delegate, self.delegate_delete)
	}
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QnnDelegateApiVersion {
	major: u32,
	minor: u32,
	patch: u32,
}
impl Display for QnnDelegateApiVersion {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
	}
}
/// Qnn Delegate API Version at the time of this rust porting
pub const RS_PORT_QNN_DELEGATE_API_VERSION: QnnDelegateApiVersion = QnnDelegateApiVersion {
	major: 0,
	minor: 24,
	patch: 0,
};

#[repr(C)]
pub enum TfLiteQnnDelegateBackendType {
	Undefined = 0,
	Gpu,
	Htp,
	Dsp,
	Ir,
}

#[repr(C)]
pub enum TfLiteQnnDelegateLogLevel {
	Off = 0,
	Error = 1,
	Warn = 2,
	Info = 3,
	Verbose = 4,
	Debug = 5,
}

#[repr(C)]
pub enum TfLiteQnnDelegateGraphPriority {
	Default = 0,
	Low,
	Normal,
	NormalHigh,
	High,
	Undefined,
}

#[repr(C)]
pub enum TfLiteQnnDelegateProfilingOptions {
	ProfilingOff = 0,
	BasicProfiling,
	PerOpProfiling,
	LintingProfiling,
}

#[repr(C)]
pub enum TfLiteQnnDelegateGpuPrecision {
	UserProvided = 0,
	Fp32,
	Fp16,
	Hybrid,
}

#[repr(C)]
pub enum TfLiteQnnDelegateGpuPerformanceMode {
	Default = 0,
	High,
	Normal,
	Low,
}

#[repr(C)]
pub enum TfLiteQnnDelegateHtpPerformanceMode {
	Default = 0,
	SustainedHighPerformance = 1,
	Burst = 2,
	HighPerformance = 3,
	PowerSaver = 4,
	LowPowerSaver = 5,
	HighPowerSaver = 6,
	LowBalanced = 7,
	Balanced = 8,
	ExtremePowerSaver = 9,
}

#[repr(C)]
pub enum TfLiteQnnDelegateDspPerformanceMode {
	Default = 0,
	SustainedHighPerformance = 1,
	Burst = 2,
	HighPerformance = 3,
	PowerSaver = 4,
	LowPowerSaver = 5,
	HighPowerSaver = 6,
	LowBalanced = 7,
	Balanced = 8,
}

#[repr(C)]
pub enum TfLiteQnnDelegateHtpPerfCtrlStrategy {
	Manual = 0,
	Auto = 1,
}

#[repr(C)]
pub enum TfLiteQnnDelegateDspPerfCtrlStrategy {
	Manual = 0,
	Auto = 1,
}

#[repr(C)]
pub enum TfLiteQnnDelegateDspPdSession {
	Unsigned = 0,
	Signed,
	Adaptive,
}

#[repr(C)]
pub enum TfLiteQnnDelegateDspEncoding {
	Static = 0,
	Dynamic = 1,
	Unknown = 0x7fffffff,
}

#[repr(C)]
pub enum TfLiteQnnDelegateHtpPdSession {
	Unsigned = 0,
	Signed,
}

#[repr(C)]
pub enum TfLiteQnnDelegateHtpPrecision {
	Quantized = 0,
	Fp16,
}

#[repr(C)]
pub enum TfLiteQnnDelegateHtpOptimizationStrategy {
	OptimizeForInference = 0,
	OptimizeForPrepare,
	OptimizeForInferenceO3,
}

#[repr(C)]
pub enum TfLiteQnnDelegatePerformanceAction {
	PerformanceVote = 0,
	PerformanceRelease = 1,
}

#[repr(C)]
pub struct TfLiteQnnDelegateGpuBackendOptions {
	precision: TfLiteQnnDelegateGpuPrecision,
	performance_mode: TfLiteQnnDelegateGpuPerformanceMode,
	kernel_repo_dir: *const std::os::raw::c_char,
}

#[repr(C)]
pub struct TfLiteQnnDelegateHtpBackendOptions {
	pub performance_mode: TfLiteQnnDelegateHtpPerformanceMode,
	pub perf_ctrl_strategy: TfLiteQnnDelegateHtpPerfCtrlStrategy,
	pub precision: TfLiteQnnDelegateHtpPrecision,
	pub pd_session: TfLiteQnnDelegateHtpPdSession,
	pub optimization_strategy: TfLiteQnnDelegateHtpOptimizationStrategy,
	pub use_conv_hmx: bool,
	pub use_fold_relu: bool,
	pub vtcm_size: u32,
	pub num_hvx_threads: u32,
	pub device_id: u32,
}

#[repr(C)]
pub struct TfLiteQnnDelegateDspBackendOptions {
	performance_mode: TfLiteQnnDelegateDspPerformanceMode,
	perf_ctrl_strategy: TfLiteQnnDelegateDspPerfCtrlStrategy,
	pd_session: TfLiteQnnDelegateDspPdSession,
	encoding: TfLiteQnnDelegateDspEncoding,
}

#[repr(C)]
pub struct TfLiteQnnDelegateIrBackendOptions {
	output_path: *const std::os::raw::c_char,
}

#[repr(C)]
pub struct TfLiteQnnDelegateOpPackageOpMap {
	custom_op_name: *const std::os::raw::c_char,
	qnn_op_type_name: *const std::os::raw::c_char,
}

#[repr(C)]
pub struct TfLiteQnnDelegateOpPackageInfo {
	op_package_name: *const std::os::raw::c_char,
	op_package_path: *const std::os::raw::c_char,
	interface_provider: *const std::os::raw::c_char,
	target: *const std::os::raw::c_char,
	num_ops_map: i32,
	ops_map: *mut TfLiteQnnDelegateOpPackageOpMap,
}

#[repr(C)]
pub struct TfLiteQnnDelegateOpPackageOptions {
	num_op_package_infos: i32,
	op_package_index: *mut TfLiteQnnDelegateOpPackageInfo,
}

#[repr(C)]
pub struct TfLiteQnnDelegateSkipOption {
	skip_delegate_ops: *const i32,
	skip_delegate_ops_nr: u32,
	skip_delegate_node_ids: *const i32,
	skip_delegate_node_ids_nr: u32,
}

#[repr(C)]
pub struct TfLiteQnnDelegateOptions {
	pub backend_type: TfLiteQnnDelegateBackendType,
	pub library_path: *const std::os::raw::c_char,
	pub skel_library_path: *const std::os::raw::c_char,
	pub gpu_options: TfLiteQnnDelegateGpuBackendOptions,
	pub htp_options: TfLiteQnnDelegateHtpBackendOptions,
	pub dsp_options: TfLiteQnnDelegateDspBackendOptions,
	pub ir_options: TfLiteQnnDelegateIrBackendOptions,
	pub log_level: TfLiteQnnDelegateLogLevel,
	pub profiling: TfLiteQnnDelegateProfilingOptions,
	pub op_package_options: TfLiteQnnDelegateOpPackageOptions,
	pub tensor_dump_output_path: *const std::os::raw::c_char,
	pub cache_dir: *const std::os::raw::c_char,
	pub model_token: *const std::os::raw::c_char,
	pub skip_option: TfLiteQnnDelegateSkipOption,
	pub graph_priority: TfLiteQnnDelegateGraphPriority,
}

#[repr(i32)]
pub enum TfLiteQnnDelegateCapabilityStatus {
	NotSupported = 0,
	Supported = 1,
}

#[repr(C)]
pub enum TfLiteQnnDelegateCapability {
	HtpRuntimeQuant = 0,
	HtpRuntimeFp16 = 1,
	GpuRuntime = 2,
	DspRuntime = 3,
}

pub type TfLiteQnnDelegateOptionsDefaultF = unsafe extern "C" fn() -> TfLiteQnnDelegateOptions;

pub type TfLiteQnnDelegateCreateF =
	unsafe extern "C" fn(options: *const TfLiteQnnDelegateOptions) -> *mut sys::TfLiteDelegate;

pub type TfLiteQnnDelegateDeleteF = unsafe extern "C" fn(delegate: *mut sys::TfLiteDelegate);

pub type TfLiteQnnDelegateSetPerfF = unsafe extern "C" fn(
	delegate: *mut sys::TfLiteDelegate,
	action: TfLiteQnnDelegatePerformanceAction,
);

pub type TfLiteQnnDelegateHasCapabilityF =
	unsafe extern "C" fn(
		capability: TfLiteQnnDelegateCapability,
	) -> TfLiteQnnDelegateCapabilityStatus;

pub type TfLiteQnnDelegateUpdateHtpPerfModeF = unsafe extern "C" fn(
	delegate: *mut sys::TfLiteDelegate,
	mode: TfLiteQnnDelegateHtpPerformanceMode,
) -> bool;

pub type TfLiteQnnDelegateUpdateDspPerfModeF = unsafe extern "C" fn(
	delegate: *mut sys::TfLiteDelegate,
	mode: TfLiteQnnDelegateDspPerformanceMode,
) -> bool;

pub type TfLiteQnnDelegateGetApiVersionF = unsafe extern "C" fn() -> QnnDelegateApiVersion;

pub type TfLiteQnnDelegateAllocCustomMemF =
	unsafe extern "C" fn(bytes: usize, alignment: usize) -> *mut std::os::raw::c_void;

pub type TfLiteQnnDelegateFreeCustomMemF =
	unsafe extern "C" fn(buffer_ptr: *mut std::os::raw::c_void);

#[repr(C)]
pub struct TfLiteQnnDelegateProfilingResult {
	buffer: *const u8,
	buffer_length: u32,
}

pub type TfLiteQnnDelegateGetProfilingResultF =
	unsafe extern "C" fn(delegate: *mut sys::TfLiteDelegate) -> TfLiteQnnDelegateProfilingResult;

pub type TfLiteQnnDelegateClearProfilingResultF =
	unsafe extern "C" fn(delegate: *mut sys::TfLiteDelegate);
