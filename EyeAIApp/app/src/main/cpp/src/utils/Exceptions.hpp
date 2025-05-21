#pragma once

#include <exception>
#include <format>
#include <stdexcept>

class FormatNotRGBA888Exception : public std::runtime_error {
  public:
	explicit FormatNotRGBA888Exception(int32_t actual_format)
		: std::runtime_error(
			  std::format("format is {} instead of RGBA8888!", actual_format)
		  ) {}
};

class FailedToLockPixelsException : public std::exception {
  public:
	[[nodiscard]] const char* what() const noexcept override {
		return "failed to lock pixels of bitmap";
	}
};