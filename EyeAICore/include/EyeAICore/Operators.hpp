#pragma once

#include <span>
#include <string>
#include <tl/expected.hpp>

class Operator {
  public:
	Operator() = default;
	Operator(const Operator&) = default;
	Operator(Operator&&) = default;
	Operator& operator=(const Operator&) = default;
	Operator& operator=(Operator&&) = default;
	virtual ~Operator() = default;

	[[nodiscard]] virtual tl::expected<void, std::string>
	execute(std::span<float> input) const = 0;
};

/// rescales values from [min, max] to [0, 1]
class MinMaxOperator : public Operator {
  public:
	[[nodiscard]] tl::expected<void, std::string>
	execute(std::span<float> values) const override;
};

/// normalizes rgb input values (3 floats for r, g and b) based on their mean
/// and standard deviation values
class RgbNormalizeOperator : public Operator {
  public:
	std::array<float, 3> mean = {123.675f, 116.28f, 103.53f};
	std::array<float, 3> stddev = {58.395f, 57.12f, 57.375f};

	[[nodiscard]] tl::expected<void, std::string>
	execute(std::span<float> values) const override;
};