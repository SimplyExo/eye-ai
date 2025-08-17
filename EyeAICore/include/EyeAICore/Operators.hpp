#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

enum class FloatTensorFormat : std::uint8_t {
	/// 3 floats for r, g, b, all in range [0.f, 255.f]
	ImageRGB255Float,
	/// 3 floats for r, g, b, with mean of {123.675f, 116.28f, 103.53f} and
	/// standard deviation of {58.395f, 57.12f, 57.375f}
	MiDaSImageRGBFloat,
	/// 3 floats for r, g, b, all in range [0.f, 1.f]
	YoloImageRGBFloat,
	/// 1 float per pixel for relative depth in range [0.f, 1.f]
	RelativeDepth,
	/// 1 float per pixel for relative depth, all values possible
	RawRelativeDepth,
	/// special yolo output of detected objects and their confidence
	YoloOutput
};

std::string_view format_float_tensor_format(FloatTensorFormat format);

struct [[nodiscard]] OperatorError {
	std::string error_msg;

	explicit OperatorError(std::string&& error_msg)
		: error_msg(std::move(error_msg)) {}

	[[nodiscard]] std::string to_string() const { return error_msg; }
	bool operator==(const OperatorError& other) const = default;
};

/// Abstract base class that simply requires a virtual execute method to be
/// implemented. This class is only meant for internal runtime invoking.
class OperatorBase {
  public:
	OperatorBase() = default;
	OperatorBase(const OperatorBase&) = default;
	OperatorBase(OperatorBase&&) = default;
	OperatorBase& operator=(const OperatorBase&) = default;
	OperatorBase& operator=(OperatorBase&&) = default;
	virtual ~OperatorBase() = default;

	[[nodiscard]] virtual FloatTensorFormat get_input_format() const = 0;

	[[nodiscard]] virtual FloatTensorFormat get_output_format() const = 0;

	/// Will be called in @ref TfLiteRuntime either before or after the
	/// inference, input is guaranteed to be format of get_input_format()
	[[nodiscard]] virtual std::optional<OperatorError>
	execute(std::span<float> input) const = 0;
};

/// Abstract base class for all input/output operators that modify a float
/// array. Matching input/output formats are validated at comp time in the
/// OperatorChain.
template<FloatTensorFormat Input, FloatTensorFormat Output>
class Operator : public OperatorBase {
  public:
	Operator() = default;
	Operator(const Operator&) = default;
	Operator(Operator&&) noexcept = default;
	Operator& operator=(const Operator&) = default;
	Operator& operator=(Operator&&) noexcept = default;
	~Operator() override = default;

	constexpr static FloatTensorFormat INPUT = Input;
	constexpr static FloatTensorFormat OUTPUT = Output;

	[[nodiscard]] FloatTensorFormat get_input_format() const override {
		return INPUT;
	}

	[[nodiscard]] FloatTensorFormat get_output_format() const override {
		return OUTPUT;
	}
};

/// checks if a list of operators form a valid chain of matching input/output
/// formats
///
/// for example:
///
/// valid: Format::A -> Format::B, Format::B -> Format::C
///
/// invalid: Format::A -> Format::B, Format::C -> Format::D
template<typename... Ops>
static constexpr bool check_op_chain() {
	if constexpr (sizeof...(Ops) < 2) {
		return true;
	} else {
		using Tuple = std::tuple<Ops...>;

		return []<std::size_t... Is>(std::index_sequence<Is...>) {
			return (
				(std::tuple_element_t<Is, Tuple>::OUTPUT ==
				 std::tuple_element_t<Is + 1, Tuple>::INPUT) &&
				...
			);
		}(std::make_index_sequence<sizeof...(Ops) - 1>{});
	}
}

static_assert(check_op_chain<
			  Operator<
				  FloatTensorFormat::ImageRGB255Float,
				  FloatTensorFormat::MiDaSImageRGBFloat>,
			  Operator<
				  FloatTensorFormat::MiDaSImageRGBFloat,
				  FloatTensorFormat::YoloImageRGBFloat>>());

static_assert(not check_op_chain<
			  Operator<
				  FloatTensorFormat::ImageRGB255Float,
				  FloatTensorFormat::MiDaSImageRGBFloat>,
			  Operator<
				  FloatTensorFormat::YoloImageRGBFloat,
				  FloatTensorFormat::MiDaSImageRGBFloat>>());

template<typename... Ops>
class OperatorChain {
  public:
	explicit OperatorChain(Ops... ops) : operators(std::move(ops)...) {}

	/// returns a vector of unique_ptr<OperatorBase>, such that they can be used
	/// at runtime without having to know the exact types
	[[nodiscard]] constexpr std::vector<std::unique_ptr<OperatorBase>>
	to_runtime_base() && {
		std::vector<std::unique_ptr<OperatorBase>> result;
		result.reserve(sizeof...(Ops));

		auto convert = [&result](auto&&... args) {
			(result.emplace_back(
				 std::make_unique<std::decay_t<decltype(args)>>(
					 std::forward<decltype(args)>(args)
				 )
			 ),
			 ...);
		};

		std::apply(convert, std::move(operators));
		return result;
	}

  private:
	std::tuple<Ops...> operators;

	static_assert(
		check_op_chain<Ops...>(),
		"each operator's output format in the chain needs to match the next "
		"operator's input format (see check_op_chain)"
	);
};

/// rescales values from [min, max] to [0, 1]
class RelativeDepthPostOperator : public Operator<
									  FloatTensorFormat::RawRelativeDepth,
									  FloatTensorFormat::RelativeDepth> {
  public:
	[[nodiscard]] std::optional<OperatorError>
	execute(std::span<float> input) const override;
};

/// normalizes rgb input values (3 floats for r, g and b) based on their mean
/// and standard deviation values
class MiDaSImageOperator : public Operator<
							   FloatTensorFormat::ImageRGB255Float,
							   FloatTensorFormat::MiDaSImageRGBFloat> {
  public:
	[[nodiscard]] std::optional<OperatorError>
	execute(std::span<float> input) const override;

  private:
	constexpr static std::array<float, 3> MEAN = {123.675f, 116.28f, 103.53f};
	constexpr static std::array<float, 3> STDDEV = {58.395f, 57.12f, 57.375f};
};

class YoloImageOperator : public Operator<
							  FloatTensorFormat::ImageRGB255Float,
							  FloatTensorFormat::YoloImageRGBFloat> {
  public:
	[[nodiscard]] std::optional<OperatorError>
	execute(std::span<float> input) const override;
};