#pragma once

#include <format>
#include <string>
#include <tl/expected.hpp>
#include <variant>

namespace tl {
/** same as tl::unexpected, but with formatted error using std::format */
template<typename... Args>
[[nodiscard]] unexpected<std::string>
unexpected_fmt(const std::format_string<Args...> fmt, Args&&... args) {
	return unexpected(std::vformat(fmt.get(), std::make_format_args(args...)));
}
} // namespace tl

template<typename... Ts>
struct Overloads : Ts... {
	using Ts::operator()...;
};

template<typename... Ts>
Overloads(Ts...) -> Overloads<Ts...>;

template<typename... Ts>
struct [[nodiscard]] CombinedError : public std::variant<Ts...> {
	using std::variant<Ts...>::variant;
	using std::variant<Ts...>::operator=;
	using std::variant<Ts...>::swap;

	[[nodiscard]] std::string to_string() const {
		return std::visit(
			[](const auto& ts) -> std::string { return ts.to_string(); }, *this
		);
	}

	template<typename... Fs>
	auto match(Fs&&... fs) const& {
		const auto visitor = Overloads(std::forward<Fs>(fs)...);

		return std::visit(visitor, *this);
	}

	template<typename... Fs>
	auto match(Fs&&... fs) & {
		const auto visitor = Overloads(std::forward<Fs>(fs)...);

		return std::visit(visitor, *this);
	}

	template<typename... Fs>
	auto match(Fs&&... fs) && {
		const auto visitor = Overloads(std::forward<Fs>(fs)...);

		return std::visit(visitor, std::move(*this));
	}
};

#define COMBINED_ERROR(name, ...)                                              \
	struct [[nodiscard]] name : public CombinedError<__VA_ARGS__> {            \
		using CombinedError::CombinedError;                                    \
		using CombinedError::operator=;                                        \
		using CombinedError::swap;                                             \
		using CombinedError::match;                                            \
		using CombinedError::to_string;                                        \
	}
