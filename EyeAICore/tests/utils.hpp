#pragma once

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include <filesystem>
#include <format>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <npy.hpp>
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <tl/expected.hpp>

template<typename E>
std::string error_to_string(const E& e);

template<Error E>
std::string error_to_string(const E& e) {
	return e.to_string();
}

template<>
inline std::string error_to_string<std::string>(const std::string& e) {
	return e;
}

template<typename T, typename E>
static void expect_result_has_value(const tl::expected<T, E>& result) {
	EXPECT_TRUE(result.has_value())
		<< "Expected result to have value, but it has error: "
		<< error_to_string(result.error());
}

#define EXPECT_RESULT_HAS_VALUE(result) expect_result_has_value(result)

template<typename T>
static void expect_no_optional_error(const std::optional<T>& optional_error) {
	if (optional_error) {
		FAIL() << "Optional error was not expected: "
			   << error_to_string(optional_error.value());
	}
}

#define EXPECT_NO_OPTIONAL_ERROR(optional_error)                               \
	expect_no_optional_error(optional_error)

tl::expected<std::vector<int8_t>, std::string>
read_model_data(const std::filesystem::path& filepath);

tl::expected<std::unique_ptr<DepthModel>, std::string>
create_test_depth_model();

template<typename... InputOps, typename... OutputOps>
static tl::expected<std::unique_ptr<TfLiteRuntime>, std::string>
create_test_tflite_runtime(
	const std::filesystem::path& model_path,
	FloatTensorFormat model_input_format,
	FloatTensorFormat model_output_format,
	OperatorChain<InputOps...>&& input_operators,
	OperatorChain<OutputOps...>&& output_operators,
	ProfilingFrame& profiling_frame
) {
	const auto model_last_modified =
		std::filesystem::last_write_time(model_path);
	const std::string model_token = std::format(
		"{}_{}", model_path.filename().string(), model_last_modified
	);

	auto model_data_result = read_model_data(model_path);
	if (!model_data_result)
		return tl::unexpected(model_data_result.error());

	const auto gpu_serialization_path =
		std::filesystem::temp_directory_path() / "EyeAICore/gpu_delegate_cache";
	std::filesystem::create_directories(gpu_serialization_path);

	auto runtime_result = TfLiteRuntime::create(
		std::move(*model_data_result), gpu_serialization_path.string(),
		model_token, model_input_format, model_output_format,
		std::move(input_operators), std::move(output_operators),
		[](const std::string msg) { std::cout << "[WARN]  " << msg << '\n'; },
		[](const std::string msg) { std::cerr << "[ERROR] " << msg << '\n'; },
		profiling_frame
	);
	if (runtime_result)
		return std::move(runtime_result.value());
	return tl::unexpected(runtime_result.error().to_string());
}

tl::expected<std::vector<float>, std::string> load_image_file(
	const std::filesystem::path& filepath,
	size_t target_width,
	size_t target_height
);

tl::expected<std::vector<std::string>, std::string>
read_coco_labels_file(const std::filesystem::path& filepath);