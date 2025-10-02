use std::sync::Arc;

use libloading::Library;

use crate::sys;

pub struct XnnPackVt {
    pub library: Arc<Library>,
    pub delegate_options_default: TfLiteXNNPackDelegateOptionsDefaultF,
    pub delegate_create: TfLiteXNNPackDelegateCreateF,
    pub delegate_delete: TfLiteXNNPackDelegateDeleteF,
}

impl XnnPackVt {
    pub fn load(library: Arc<Library>) -> Option<Self> {
        let delegate_options_default =
            unsafe { library.get(b"TfLiteXNNPackDelegateOptionsDefault\0") };

        // If the function's not in the library, it's not supported in this build.
        let delegate_options_default = if let Ok(value) = delegate_options_default {
            *value
        } else {
            return None;
        };

        let delegate_create = unsafe { *library.get(b"TfLiteXNNPackDelegateCreate\0").unwrap() };
        let delegate_delete = unsafe { *library.get(b"TfLiteXNNPackDelegateDelete\0").unwrap() };

        Some(Self {
            library,
            delegate_options_default,
            delegate_create,
            delegate_delete,
        })
    }
}

pub type TfLiteXNNPackDelegateOptionsDefaultF =
    unsafe extern "C" fn() -> sys::TfLiteXNNPackDelegateOptions;

pub type TfLiteXNNPackDelegateCreateF =
    unsafe extern "C" fn(options: *const TfLiteXNNPackDelegateOptions) -> *mut sys::TfLiteDelegate;

pub type TfLiteXNNPackDelegateDeleteF = unsafe extern "C" fn(delegate: *mut sys::TfLiteDelegate);

#[repr(C)]
pub struct TfLiteXNNPackDelegateOptions {
    pub num_threads: i32,
}
