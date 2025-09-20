#pragma once

#include "EyeAICore/TensorBuffer.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include <memory>

class DepthModel;

class MetricDepthModel {
  public:
	constexpr static size_t POLYNOMIAL_DEGREE = 4;
	constexpr static size_t COEFFS_COUNT = POLYNOMIAL_DEGREE + 1;

	using CreateResult = tl::
		expected<std::unique_ptr<MetricDepthModel>, TfLiteCreateRuntimeError>;

	[[nodiscard]] static CreateResult create(
		std::vector<int8_t>&& depth_model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view depth_model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback,
		bool enable_npu,
		std::string npu_skel_directory
	);

	using RunResult = tl::expected<
		FloatTensorBuffer<FloatTensorFormat::MetricDepth>,
		TfLiteRunInferenceError>;

	/// @param input should be 3 * width * height
	[[nodiscard]] RunResult
	run(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input);

	[[nodiscard]] std::span<const int> get_input_shape() const;

	[[nodiscard]] std::span<const int> get_output_shape() const;

	MetricDepthModel(
		std::unique_ptr<DepthModel>&& depth_model
	);

  private:
	constexpr static std::array<float, 5> REL2ABS_COEFFS = {
		4.30595f, -6.5995E-03f, 5.25059E-6f, -2.7962E-9f, 9.28594E-13f
	};

	std::unique_ptr<DepthModel> depth_model;
};

/**
 * Polynomial function of degree 4 using Horner's method.
 *
 * coeffs: {a0, a1, a2, a3, a4}
 *
 * polynomial_4(x) = a0 + a1 * x + a2 * x² + a3 * x³ + a4 * x⁴
 */
constexpr static float
polynomial_4(float x, const std::array<float, 5>& coeffs) {
	float y = coeffs[4];
	y = y * x + coeffs[3];
	y = y * x + coeffs[2];
	y = y * x + coeffs[1];
	y = y * x + coeffs[0];
	return y;
}

/**
 * @see polynomial_n
 * @return new y value
 */
template<size_t I, size_t N>
constexpr static float
polynomial_n_impl(float x, float y, const std::array<float, N + 1>& coeffs) {
	static_assert(I <= N);

	const float new_y = (I != N) ? (coeffs[I] + (y * x)) : coeffs[I];

	if constexpr (I == 0) {
		return new_y;
	} else {
		return polynomial_n_impl<I - 1, N>(x, new_y, coeffs);
	}
}

/**
 * Polynomial function of degree N using Horner's method.
 *
 * coeffs: {a0, ..., aN}
 *
 * polynomial_n(x) = a0 + a1 * x + a2 * x² + ... + aN * x^N
 *
 * polynomial_n<4> Compiles to the same as if written by hand like @ref
 * polynomial_4 with any optimization enabled
 * (tested with clang, anything other than -O0 produces the same machine code).
 */
template<size_t N>
constexpr static float
polynomial_n(float x, const std::array<float, N + 1>& coeffs) {
	return polynomial_n_impl<N, N>(x, 0.f, coeffs);
}