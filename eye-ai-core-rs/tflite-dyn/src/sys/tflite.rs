use std::{
	ffi::{c_void, CStr},
	os::raw::c_char,
	sync::Arc,
};

use libloading::Library;
use semver::{Version, VersionReq};

use crate::Error;

pub struct TfLiteVt {
	pub library: Arc<Library>,
	pub version: TfLiteVersionF,
	pub model_create: TfLiteModelCreateF,
	pub model_delete: TfLiteModelDeleteF,
	pub interpreter_options_create: TfLiteInterpreterOptionsCreateF,
	pub interpreter_options_delete: TfLiteInterpreterOptionsDeleteF,
	pub interpreter_options_set_num_threads: TfLiteInterpreterOptionsSetNumThreadsF,
	pub interpreter_options_set_error_reporter: TfLiteInterpreterOptionsSetErrorReporterF,
	pub interpreter_options_copy: TfLiteInterpreterOptionsCopyF,
	pub interpreter_options_add_delegate: TfLiteInterpreterOptionsAddDelegateF,
	pub interpreter_create: TfLiteInterpreterCreateF,
	pub interpreter_delete: TfLiteInterpreterDeleteF,
	pub interpreter_get_input_tensor_count: TfLiteInterpreterGetInputTensorCountF,
	pub interpreter_get_input_tensor: TfLiteInterpreterGetInputTensorF,
	pub interpreter_allocate_tensors: TfLiteInterpreterAllocateTensorsF,
	pub interpreter_invoke: TfLiteInterpreterInvokeF,
	pub interpreter_get_output_tensor_count: TfLiteInterpreterGetOutputTensorCountF,
	pub interpreter_get_output_tensor: TfLiteInterpreterGetOutputTensorF,
	pub tensor_type: TfLiteTensorTypeF,
	pub tensor_num_dims: TfLiteTensorNumDimsF,
	pub tensor_dim: TfLiteTensorDimF,
	pub tensor_byte_size: TfLiteTensorByteSizeF,
	pub tensor_data: TfLiteTensorDataF,
	pub tensor_name: TfLiteTensorNameF,
}

impl TfLiteVt {
	pub fn load(library: Arc<Library>) -> Result<Self, Error> {
		let version: TfLiteVersionF = unsafe { *library.get(b"TfLiteVersion\0").unwrap() };

		// Validate the DLL version is compatible with the header version we're targeting
		let version_str = unsafe { CStr::from_ptr((version)()) };
		let dll_version = Version::parse(version_str.to_str().unwrap()).unwrap();
		let target_version = VersionReq::parse("2.8.0").unwrap();
		if !target_version.matches(&dll_version) {
			return Err(Error::TfLiteApiVersionMismatch {
				expected: target_version,
				library_version: dll_version,
			});
		}

		let model_create = unsafe { *library.get(b"TfLiteModelCreate\0").unwrap() };
		let model_delete = unsafe { *library.get(b"TfLiteModelDelete\0").unwrap() };
		let interpreter_options_create =
			unsafe { *library.get(b"TfLiteInterpreterOptionsCreate\0").unwrap() };
		let interpreter_options_delete =
			unsafe { *library.get(b"TfLiteInterpreterOptionsDelete\0").unwrap() };
		let interpreter_options_set_num_threads = unsafe {
			*library
				.get(b"TfLiteInterpreterOptionsSetNumThreads\0")
				.unwrap()
		};
		let interpreter_options_set_error_reporter = unsafe {
			*library
				.get(b"TfLiteInterpreterOptionsSetErrorReporter\0")
				.unwrap()
		};
		let interpreter_options_copy =
			unsafe { *library.get(b"TfLiteInterpreterOptionsCopy\0").unwrap() };
		let interpreter_options_add_delegate = unsafe {
			*library
				.get(b"TfLiteInterpreterOptionsAddDelegate\0")
				.unwrap()
		};
		let interpreter_create = unsafe { *library.get(b"TfLiteInterpreterCreate\0").unwrap() };
		let interpreter_delete = unsafe { *library.get(b"TfLiteInterpreterDelete\0").unwrap() };
		let interpreter_get_input_tensor_count = unsafe {
			*library
				.get(b"TfLiteInterpreterGetInputTensorCount\0")
				.unwrap()
		};
		let interpreter_get_input_tensor =
			unsafe { *library.get(b"TfLiteInterpreterGetInputTensor\0").unwrap() };
		let interpreter_allocate_tensors =
			unsafe { *library.get(b"TfLiteInterpreterAllocateTensors\0").unwrap() };
		let interpreter_invoke = unsafe { *library.get(b"TfLiteInterpreterInvoke\0").unwrap() };
		let interpreter_get_output_tensor_count = unsafe {
			*library
				.get(b"TfLiteInterpreterGetOutputTensorCount\0")
				.unwrap()
		};
		let interpreter_get_output_tensor =
			unsafe { *library.get(b"TfLiteInterpreterGetOutputTensor\0").unwrap() };
		let tensor_type = unsafe { *library.get(b"TfLiteTensorType\0").unwrap() };
		let tensor_num_dims = unsafe { *library.get(b"TfLiteTensorNumDims\0").unwrap() };
		let tensor_dim = unsafe { *library.get(b"TfLiteTensorDim\0").unwrap() };
		let tensor_byte_size = unsafe { *library.get(b"TfLiteTensorByteSize\0").unwrap() };
		let tensor_data = unsafe { *library.get(b"TfLiteTensorData\0").unwrap() };
		let tensor_name = unsafe { *library.get(b"TfLiteTensorName\0").unwrap() };

		Ok(Self {
			library,
			version,
			model_create,
			model_delete,
			interpreter_options_create,
			interpreter_options_delete,
			interpreter_options_set_num_threads,
			interpreter_options_set_error_reporter,
			interpreter_options_copy,
			interpreter_options_add_delegate,
			interpreter_create,
			interpreter_delete,
			interpreter_get_input_tensor_count,
			interpreter_get_input_tensor,
			interpreter_allocate_tensors,
			interpreter_invoke,
			interpreter_get_output_tensor_count,
			interpreter_get_output_tensor,
			tensor_type,
			tensor_num_dims,
			tensor_dim,
			tensor_byte_size,
			tensor_data,
			tensor_name,
		})
	}
}

pub type TfLiteVersionF = unsafe extern "C" fn() -> *const c_char;

pub type TfLiteModelCreateF =
	unsafe extern "C" fn(model_data: *const c_void, size: usize) -> *mut TfLiteModel;

pub type TfLiteModelDeleteF = unsafe extern "C" fn(model: *mut TfLiteModel);

pub type TfLiteInterpreterOptionsCreateF = unsafe extern "C" fn() -> *mut TfLiteInterpreterOptions;

pub type TfLiteInterpreterOptionsDeleteF =
	unsafe extern "C" fn(options: *mut TfLiteInterpreterOptions);

pub type TfLiteInterpreterOptionsSetNumThreadsF =
	unsafe extern "C" fn(options: *mut TfLiteInterpreterOptions, num_threads: i32);

pub type TfLiteInterpreterOptionsAddDelegateF =
	unsafe extern "C" fn(options: *mut TfLiteInterpreterOptions, delegate: *mut TfLiteDelegate);

pub type TfLiteErrorCallbackF = unsafe extern "C" fn(
	user_data_ptr: *mut std::os::raw::c_void,
	format: *const std::os::raw::c_char,
	args: va_list::VaList,
);

pub type TfLiteInterpreterOptionsSetErrorReporterF = unsafe extern "C" fn(
	options: *mut TfLiteInterpreterOptions,
	error_callback: TfLiteErrorCallbackF,
	user_data_ptr: *mut std::os::raw::c_void,
);

pub type TfLiteInterpreterOptionsCopyF =
	unsafe extern "C" fn(option: *const TfLiteInterpreterOptions) -> *mut TfLiteInterpreterOptions;

pub type TfLiteInterpreterCreateF = unsafe extern "C" fn(
	model: *const TfLiteModel,
	options: *const TfLiteInterpreterOptions,
) -> *mut TfLiteInterpreter;

pub type TfLiteInterpreterDeleteF = unsafe extern "C" fn(interpreter: *mut TfLiteInterpreter);

pub type TfLiteInterpreterGetInputTensorCountF =
	unsafe extern "C" fn(interpreter: *const TfLiteInterpreter) -> i32;

pub type TfLiteInterpreterGetInputTensorF =
	unsafe extern "C" fn(interpreter: *const TfLiteInterpreter, index: i32) -> *mut TfLiteTensor;

pub type TfLiteInterpreterAllocateTensorsF =
	unsafe extern "C" fn(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;

pub type TfLiteInterpreterInvokeF =
	unsafe extern "C" fn(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;

pub type TfLiteInterpreterGetOutputTensorCountF =
	unsafe extern "C" fn(interpreter: *const TfLiteInterpreter) -> i32;

pub type TfLiteInterpreterGetOutputTensorF =
	unsafe extern "C" fn(interpreter: *const TfLiteInterpreter, index: i32) -> *mut TfLiteTensor;

pub type TfLiteTensorTypeF = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> TfLiteType;

pub type TfLiteTensorNumDimsF = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> i32;

pub type TfLiteTensorDimF =
	unsafe extern "C" fn(tensor: *const TfLiteTensor, dim_index: i32) -> i32;

pub type TfLiteTensorByteSizeF = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> usize;

pub type TfLiteTensorDataF = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> *const c_void;

pub type TfLiteTensorNameF = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> *const c_char;

#[repr(C)]
pub struct TfLiteModel {
	private: [u8; 0],
}

#[repr(C)]
pub struct TfLiteInterpreterOptions {
	private: [u8; 0],
}

#[repr(C)]
pub struct TfLiteDelegate {
	private: [u8; 0],
}

#[repr(C)]
pub struct TfLiteInterpreter {
	private: [u8; 0],
}

#[repr(C)]
pub struct TfLiteTensor {
	private: [u8; 0],
}

#[repr(C)]
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum TfLiteStatus {
	Ok = 0,
	Error = 1,
	DelegateError = 2,
	ApplicationError = 3,
	DelegateDataNotFound = 4,
	DelegateDataWriteError = 5,
	DelegateDataReadError = 6,
	UnresolvedOps = 7,
}

#[repr(C)]
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum TfLiteType {
	NoType = 0,
	Float32 = 1,
	Int32 = 2,
	UInt8 = 3,
	Int64 = 4,
	String = 5,
	Bool = 6,
	Int16 = 7,
	Complex64 = 8,
	Int8 = 9,
	Float16 = 10,
	Float64 = 11,
	Complex128 = 12,
	UInt64 = 13,
	Resource = 14,
	Variant = 15,
	UInt32 = 16,
}
