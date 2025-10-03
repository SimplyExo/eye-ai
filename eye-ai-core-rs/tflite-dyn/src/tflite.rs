use std::{
	ffi::{CStr, OsStr},
	fmt::{self, Debug, Formatter},
	marker::PhantomData,
	rc::Rc,
	sync::Arc,
};

use libloading::Library;

use crate::{
	sys::{self, TfLiteErrorCallbackF},
	Error, Type, XnnPackDelegateOptions,
};

pub struct TfLite {
	pub vt: Arc<sys::TfLiteVt>,
	xnnpack_vt: Option<Arc<sys::XnnPackVt>>,
}

impl TfLite {
	pub fn load<P: AsRef<OsStr>>(path: P) -> Result<Self, Error> {
		let library = Arc::new(unsafe { Library::new(path) }?);

		let vt = sys::TfLiteVt::load(library.clone())?;
		let xnnpack_vt = sys::XnnPackVt::load(library.clone());

		Ok(Self {
			vt: Arc::new(vt),
			xnnpack_vt: xnnpack_vt.map(Arc::new),
		})
	}

	pub fn version(&self) -> &CStr {
		unsafe { CStr::from_ptr((self.vt.version)()) }
	}

	pub fn model_create(&self, data: Vec<u8>) -> Result<Model, Error> {
		let model = unsafe { (self.vt.model_create)(data.as_ptr() as *const _, data.len()) };

		if model.is_null() {
			Err(Error::Generic)
		} else {
			Ok(Model {
				vt: self.vt.clone(),
				model,
				_data: data,
			})
		}
	}

	pub fn interpreter_options_create(&self) -> InterpreterOptions {
		InterpreterOptions {
			vt: self.vt.clone(),
			options: unsafe { (self.vt.interpreter_options_create)() },
			delegates: Vec::new(),
		}
	}

	pub fn interpreter_create(
		&self,
		model: Rc<Model>,
		mut options: InterpreterOptions,
	) -> Option<Interpreter> {
		let interpreter = unsafe { (self.vt.interpreter_create)(model.model, options.options) };

		if interpreter.is_null() {
			None
		} else {
			Some(Interpreter {
				vt: self.vt.clone(),
				interpreter,
				_model: model,
				_delegates: std::mem::take(&mut options.delegates),
			})
		}
	}

	pub fn xnnpack_delegate_options_default(&self) -> Option<XnnPackDelegateOptions> {
		let xnnpack_vt = self.xnnpack_vt.as_ref()?;
		Some(unsafe { (xnnpack_vt.delegate_options_default)() })
	}

	pub fn xnnpack_delegate_create(&self, options: &XnnPackDelegateOptions) -> Option<Delegate> {
		let xnnpack_vt = self.xnnpack_vt.as_ref()?;
		let delegate = unsafe { (xnnpack_vt.delegate_create)(options) };

		Some(Delegate {
			_vt: self.vt.clone(),
			delegate,
			destructor: xnnpack_vt.delegate_delete,
		})
	}
}

pub struct Model {
	vt: Arc<sys::TfLiteVt>,
	model: *mut sys::TfLiteModel,
	_data: Vec<u8>,
}

impl Drop for Model {
	fn drop(&mut self) {
		unsafe { (self.vt.model_delete)(self.model) };
	}
}

pub struct InterpreterOptions {
	vt: Arc<sys::TfLiteVt>,
	options: *mut sys::TfLiteInterpreterOptions,
	delegates: Vec<Rc<Delegate>>,
}

impl InterpreterOptions {
	pub fn set_num_threads(&mut self, num_threads: i32) {
		unsafe { (self.vt.interpreter_options_set_num_threads)(self.options, num_threads) };
	}

	/// # Safety
	/// user_data_ptr will be passed to the error reporter callback.
	pub unsafe fn set_error_reporter(
		&mut self,
		error_reporter: TfLiteErrorCallbackF,
		user_data_ptr: *mut std::os::raw::c_void,
	) {
		(self.vt.interpreter_options_set_error_reporter)(
			self.options,
			error_reporter,
			user_data_ptr,
		)
	}

	pub fn add_delegate(&mut self, delegate: Rc<Delegate>) {
		unsafe { (self.vt.interpreter_options_add_delegate)(self.options, delegate.delegate) };
		self.delegates.push(delegate);
	}
}

impl Clone for InterpreterOptions {
	fn clone(&self) -> Self {
		let cloned_options = unsafe { (self.vt.interpreter_options_copy)(self.options) };
		Self {
			vt: self.vt.clone(),
			options: cloned_options,
			delegates: self.delegates.clone(),
		}
	}
}

impl Drop for InterpreterOptions {
	fn drop(&mut self) {
		unsafe { (self.vt.interpreter_options_delete)(self.options) };
	}
}

pub struct Delegate {
	_vt: Arc<sys::TfLiteVt>,
	delegate: *mut sys::TfLiteDelegate,
	destructor: unsafe extern "C" fn(*mut sys::TfLiteDelegate),
}
impl Delegate {
	pub fn new(
		_vt: Arc<sys::TfLiteVt>,
		delegate: *mut sys::TfLiteDelegate,
		destructor: unsafe extern "C" fn(*mut sys::TfLiteDelegate),
	) -> Self {
		Self {
			_vt,
			delegate,
			destructor,
		}
	}
}
impl Drop for Delegate {
	fn drop(&mut self) {
		unsafe { (self.destructor)(self.delegate) };
	}
}

pub struct Interpreter {
	vt: Arc<sys::TfLiteVt>,
	interpreter: *mut sys::TfLiteInterpreter,
	_model: Rc<Model>,
	_delegates: Vec<Rc<Delegate>>,
}

impl Interpreter {
	pub fn input_tensor_count(&self) -> i32 {
		unsafe { (self.vt.interpreter_get_input_tensor_count)(self.interpreter) }
	}

	pub fn input_tensor(&self, index: i32) -> Option<Tensor<'_>> {
		let tensor = unsafe { (self.vt.interpreter_get_input_tensor)(self.interpreter, index) };

		if tensor.is_null() {
			None
		} else {
			Some(Tensor {
				vt: self.vt.clone(),
				tensor,
				_p: PhantomData,
			})
		}
	}

	pub fn allocate_tensors(&mut self) -> Result<(), Error> {
		let status = unsafe { (self.vt.interpreter_allocate_tensors)(self.interpreter) };

		if status == sys::TfLiteStatus::Ok {
			Ok(())
		} else {
			Err(Error::ErrorStatus(status))
		}
	}

	pub fn invoke(&mut self) -> Result<(), Error> {
		let status = unsafe { (self.vt.interpreter_invoke)(self.interpreter) };

		if status == sys::TfLiteStatus::Ok {
			Ok(())
		} else {
			Err(Error::ErrorStatus(status))
		}
	}

	pub fn output_tensor_count(&self) -> i32 {
		unsafe { (self.vt.interpreter_get_output_tensor_count)(self.interpreter) }
	}

	pub fn output_tensor(&self, index: i32) -> Option<Tensor<'_>> {
		let tensor = unsafe { (self.vt.interpreter_get_output_tensor)(self.interpreter, index) };

		if tensor.is_null() {
			None
		} else {
			Some(Tensor {
				vt: self.vt.clone(),
				tensor,
				_p: PhantomData,
			})
		}
	}
}

impl Drop for Interpreter {
	fn drop(&mut self) {
		unsafe { (self.vt.interpreter_delete)(self.interpreter) };
	}
}

pub struct Tensor<'a> {
	vt: Arc<sys::TfLiteVt>,
	tensor: *mut sys::TfLiteTensor,
	_p: PhantomData<&'a ()>,
}

impl<'a> Tensor<'a> {
	pub fn type_(&self) -> Type {
		unsafe { (self.vt.tensor_type)(self.tensor) }
	}

	pub fn num_dims(&self) -> i32 {
		unsafe { (self.vt.tensor_num_dims)(self.tensor) }
	}

	pub fn dim(&self, index: i32) -> i32 {
		unsafe { (self.vt.tensor_dim)(self.tensor, index) }
	}

	pub fn data(&self) -> Option<&'a [u8]> {
		unsafe {
			let data = (self.vt.tensor_data)(self.tensor) as *const u8;
			if data.is_null() {
				return None;
			}

			let len = (self.vt.tensor_byte_size)(self.tensor);
			Some(std::slice::from_raw_parts(data, len))
		}
	}

	pub fn data_mut(&mut self) -> Option<&'a mut [u8]> {
		unsafe {
			let data = (self.vt.tensor_data)(self.tensor) as *mut u8;
			if data.is_null() {
				return None;
			}

			let len = (self.vt.tensor_byte_size)(self.tensor);
			Some(std::slice::from_raw_parts_mut(data, len))
		}
	}

	pub fn name(&self) -> &CStr {
		unsafe { CStr::from_ptr((self.vt.tensor_name)(self.tensor)) }
	}
}

impl<'a> Debug for Tensor<'a> {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		write!(
			f,
			"Tensor {{ name: {:?}, type: {:?}, dims: [",
			self.name(),
			self.type_()
		)?;

		for i in 0..self.num_dims() {
			write!(f, "{}", self.dim(i))?;
			if i < self.num_dims() - 1 {
				write!(f, ", ")?;
			}
		}

		write!(f, "] }}")
	}
}
