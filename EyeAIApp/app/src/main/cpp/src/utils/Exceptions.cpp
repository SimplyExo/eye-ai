#include "Exceptions.hpp"
#include <format>

FormatNotRGBA888Exception::FormatNotRGBA888Exception(int32_t actual_format)
	: std::runtime_error(
		  std::format("format is {} instead of RGBA8888!", actual_format)
	  ) {}