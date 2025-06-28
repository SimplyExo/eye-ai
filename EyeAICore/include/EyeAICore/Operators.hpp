#pragma once

#include <format>
#include <optional>
#include <span>
#include <string>

struct [[nodiscard]] OperatorError {
	std::string error_msg;

	[[nodiscard]] std::string to_string() const { return error_msg; }

	template<typename... Args>
	[[nodiscard]] static OperatorError
	fmt(const std::format_string<Args...> fmt, Args&&... args) {
		return OperatorError(
			std::vformat(fmt.get(), std::make_format_args(args...))
		);
	}
};

class Operator {
  public:
	Operator() = default;
	Operator(const Operator&) = default;
	Operator(Operator&&) = default;
	Operator& operator=(const Operator&) = default;
	Operator& operator=(Operator&&) = default;
	virtual ~Operator() = default;

	[[nodiscard]] virtual std::optional<OperatorError>
	execute(std::span<float> input) const = 0;
};

/// rescales values from [min, max] to [0, 1]
class MinMaxOperator : public Operator {
  public:
	[[nodiscard]] std::optional<OperatorError>
	execute(std::span<float> values) const override;
};

/// normalizes rgb input values (3 floats for r, g and b) based on their mean
/// and standard deviation values
class RgbNormalizeOperator : public Operator {
  public:
	std::array<float, 3> mean = {123.675f, 116.28f, 103.53f};
	std::array<float, 3> stddev = {58.395f, 57.12f, 57.375f};

	[[nodiscard]] std::optional<OperatorError>
	execute(std::span<float> values) const override;
};