/// This file tests some invalid TfLiteRuntime operator combinations, where the
/// input/output format of the tensor do not match.

#include "EyeAICore/Operators.hpp"
#include "utils.hpp"

TEST(TfLiteRuntimeTests, NoOperatorsFormatMismatch) {
	const std::filesystem::path midas_model_path =
		"../../EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite";

	// no operators to convert input/output formats
	auto runtime_result = create_test_tflite_runtime(
		midas_model_path, FloatTensorFormat::MiDaSImageRGBFloat,
		FloatTensorFormat::RawRelativeDepth, OperatorChain{}, OperatorChain{}
	);
	EXPECT_RESULT_HAS_VALUE(runtime_result);
	auto& runtime = *runtime_result;

	size_t input_size = 1;
	for (size_t dim : runtime->get_input_shape())
		input_size *= dim;
	size_t output_size = 1;
	for (size_t dim : runtime->get_output_shape())
		output_size *= dim;

	std::vector<float> input(input_size);
	std::vector<float> output(output_size);

	auto run_error = runtime->run_inference(
		input, FloatTensorFormat::ImageRGB255Float, output,
		FloatTensorFormat::RelativeDepth
	);
	// runtime should not run successfully: missing operators to convert
	// input/output formats
	TfLiteRunInferenceError expected_error = InvalidInputFormatForModel{
		.provided = FloatTensorFormat::ImageRGB255Float,
		.expected = FloatTensorFormat::MiDaSImageRGBFloat
	};
	EXPECT_EQ(*run_error, expected_error);
}

TEST(TfLiteRuntimeTests, CorrectOperatorChain) {
	const std::filesystem::path midas_model_path =
		"../../EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite";

	auto runtime_result = create_test_tflite_runtime(
		midas_model_path, FloatTensorFormat::MiDaSImageRGBFloat,
		FloatTensorFormat::RawRelativeDepth,
		OperatorChain{MiDaSImageOperator{}},
		OperatorChain{RelativeDepthPostOperator{}}
	);
	EXPECT_RESULT_HAS_VALUE(runtime_result);
	auto& runtime = *runtime_result;

	size_t input_size = 1;
	for (size_t dim : runtime->get_input_shape())
		input_size *= dim;
	size_t output_size = 1;
	for (size_t dim : runtime->get_output_shape())
		output_size *= dim;

	std::vector<float> input(input_size);
	std::vector<float> output(output_size);

	auto run_error = runtime->run_inference(
		input, FloatTensorFormat::ImageRGB255Float, output,
		FloatTensorFormat::RelativeDepth
	);
	if (run_error) {
		FAIL() << run_error->to_string();
	}
}

// OperatorChain{MiDaSImageOperator{}, MiDaSImageOperator{}} should throw a
// static_assert
static_assert(not check_op_chain<MiDaSImageOperator, MiDaSImageOperator>());