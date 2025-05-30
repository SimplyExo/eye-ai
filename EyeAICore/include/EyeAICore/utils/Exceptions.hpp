#pragma once

#include <exception>
#include <stdexcept>

class FormatNotRGBA888Exception : public std::runtime_error {
  public:
	explicit FormatNotRGBA888Exception(int32_t actual_format);
};

class FailedToLockPixelsException : public std::exception {
  public:
	[[nodiscard]] const char* what() const noexcept override {
		return "failed to lock pixels of bitmap";
	}
};