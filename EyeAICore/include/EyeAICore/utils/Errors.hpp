#pragma once

#include <format>
#include <string>
#include <tl/expected.hpp>
#include <variant>

namespace tl {
// when using std::make_format_args, std::forward should not be used
// NOLINTBEGIN(cppcoreguidelines-missing-std-forward)

/// same as tl::unexpected, but with formatted error using std::format
template<typename... Args>
[[nodiscard]] unexpected<std::string>
unexpected_fmt(const std::format_string<Args...> fmt, Args&&... args) {
	return unexpected(std::vformat(fmt.get(), std::make_format_args(args...)));
}

// NOLINTEND(cppcoreguidelines-missing-std-forward)
} // namespace tl

template<typename T>
concept HasEqualTo = requires(const T& a, const T& b) {
	{ a.operator==(b) } -> std::same_as<bool>;
};

/// An error is simply a value that can be formatted by calling to_string()
template<typename E>
concept Error = requires(E error) { error.to_string(); } && HasEqualTo<E>;

template<typename... Ts>
struct Overloads : Ts... {
	using Ts::operator()...;
};

template<typename... Ts>
Overloads(Ts...) -> Overloads<Ts...>;

/// Rust Enum style errors as values helper class, see @ref Error concept
template<Error... Ts>
struct [[nodiscard]] CombinedError : public std::variant<Ts...> {
	using std::variant<Ts...>::variant;
	using std::variant<Ts...>::operator=;
	using std::variant<Ts...>::swap;

	/// Formats every underlying @ref Error variant
	[[nodiscard]] std::string to_string() const {
		return std::visit(
			[](const auto& ts) -> std::string { return ts.to_string(); }, *this
		);
	}

	/// Match on every underlying @ref Error variant
	template<typename... Fs>
	auto match(Fs&&... fs) const& {
		const auto visitor = Overloads(std::forward<Fs>(fs)...);

		return std::visit(visitor, *this);
	}

	/// Match on every underlying @ref Error variant
	template<typename... Fs>
	auto match(Fs&&... fs) & {
		const auto visitor = Overloads(std::forward<Fs>(fs)...);

		return std::visit(visitor, *this);
	}

	/// Match on every underlying @ref Error variant
	template<typename... Fs>
	auto match(Fs&&... fs) && {
		const auto visitor = Overloads(std::forward<Fs>(fs)...);

		return std::visit(visitor, std::move(*this));
	}

	constexpr bool operator==(const CombinedError& other) const {
		return static_cast<const std::variant<Ts...>&>(*this) ==
			   static_cast<const std::variant<Ts...>&>(other);
	}

	constexpr bool operator!=(const CombinedError& other) const {
		return static_cast<const std::variant<Ts...>&>(*this) !=
			   static_cast<const std::variant<Ts...>&>(other);
	}
};

#define COMBINED_ERROR(name, ...)                                              \
	struct [[nodiscard]] name : public CombinedError<__VA_ARGS__> {            \
		using CombinedError::CombinedError;                                    \
		using CombinedError::operator=;                                        \
		using CombinedError::operator==;                                       \
		using CombinedError::operator!=;                                       \
		using CombinedError::swap;                                             \
		using CombinedError::match;                                            \
		using CombinedError::to_string;                                        \
	}
