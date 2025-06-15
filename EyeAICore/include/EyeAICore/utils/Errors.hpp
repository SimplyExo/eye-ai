#pragma once

#include <format>
#include <string>
#include <tl/expected.hpp>

namespace tl {
/** same as tl::unexpected, but with formatted error using std::format */
template<typename... Args>
[[nodiscard]] unexpected<std::string>
unexpected_fmt(const std::format_string<Args...> fmt, Args&&... args) {
	return unexpected(std::vformat(fmt.get(), std::make_format_args(args...)));
}
} // namespace tl