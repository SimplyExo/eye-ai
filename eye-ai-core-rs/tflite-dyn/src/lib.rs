pub mod sys;
mod tflite;

use thiserror::Error;

pub use self::tflite::*;

type Type = sys::TfLiteType;
type XnnPackDelegateOptions = sys::TfLiteXNNPackDelegateOptions;

#[derive(Debug, Clone, Copy, Error)]
pub enum Error {
	#[error("Failed to load tflite library")]
	FailedToLoad,
	#[error("Generic error")]
	Generic,
	#[error("Error status: {0:?}")]
	ErrorStatus(sys::TfLiteStatus),
}
