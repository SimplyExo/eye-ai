#include "EyeAICore/TensorBuffer.hpp"

std::string_view format_float_tensor_format(FloatTensorFormat format) {
	switch (format) {
	case FloatTensorFormat::ImageRGB:
		return "ImageRGB";
	case FloatTensorFormat::ImageRGB255:
		return "ImageRGB255";
	case FloatTensorFormat::MiDaSImageRGB:
		return "MiDaSImageRGB";
	case FloatTensorFormat::YoloImageRGB:
		return "YoloImageRGB";
	case FloatTensorFormat::RelativeDepth:
		return "RelativeDepth";
	case FloatTensorFormat::RawRelativeDepth:
		return "RawRelativeDepth";
	case FloatTensorFormat::YoloOutput:
		return "YoloOutput";
	default:
		return "invalid";
	}
}