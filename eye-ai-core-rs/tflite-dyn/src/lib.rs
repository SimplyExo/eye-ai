pub mod sys;
mod tflite;

use thiserror::Error;

pub use self::tflite::*;

type Type = sys::TfLiteType;
type XnnPackDelegateOptions = sys::TfLiteXNNPackDelegateOptions;

#[derive(Debug, Error)]
pub enum Error {
	#[error("Failed to load tflite library: {0}")]
	FailedToLoad(#[from] libloading::Error),
	#[error("TfLite API version mismatch: expected {expected}, got {library_version}")]
	TfLiteApiVersionMismatch {
		expected: semver::VersionReq,
		library_version: semver::Version,
	},
	#[error("Generic error")]
	Generic,
	#[error("Error status: {0:?}")]
	ErrorStatus(sys::TfLiteStatus),
}
