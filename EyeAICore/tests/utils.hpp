#pragma once

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
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

#define EXPECT_RESULT_HAS_VALUE(result)                                        \
	EXPECT_TRUE((result).has_value())                                          \
		<< "Expected result to have value, but it has error: "                 \
		<< error_to_string((result).error())

static tl::expected<std::vector<int8_t>, std::string>
read_model_data(const std::filesystem::path& filepath) {
	std::ifstream file(filepath, std::ios::binary | std::ios::ate);

	if (!file.is_open())
		return tl::unexpected_fmt("Failed to open file: {}", filepath.string());

	std::streamsize binary_size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<int8_t> buffer(binary_size);

	if (!file.read(reinterpret_cast<char*>(buffer.data()), binary_size))
		return tl::unexpected_fmt("Failed to read file: {}", filepath.string());

	return buffer;
}

static tl::expected<std::unique_ptr<DepthModel>, std::string>
create_test_depth_model() {
	const std::filesystem::path midas_model_path =
		"../../EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite";
	const auto midas_model_last_modified =
		std::filesystem::last_write_time(midas_model_path);
	const std::string midas_model_token = std::format(
		"{}_{}", midas_model_path.filename().string(), midas_model_last_modified
	);

	auto model_data_result = read_model_data(midas_model_path);
	if (!model_data_result)
		return tl::unexpected(model_data_result.error());

	const auto gpu_serialization_path =
		std::filesystem::temp_directory_path() / "EyeAICore/gpu_delegate_cache";
	std::filesystem::create_directories(gpu_serialization_path);

	auto depth_model_result = DepthModel::create(
		std::move(*model_data_result), gpu_serialization_path.string(),
		midas_model_token,
		[](const std::string msg) { std::cout << "[WARN]  " << msg << '\n'; },
		[](const std::string msg) { std::cerr << "[ERROR] " << msg << '\n'; }
	);
	if (depth_model_result)
		return std::move(depth_model_result.value());
	return tl::unexpected(depth_model_result.error().to_string());
}

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

static tl::expected<std::vector<float>, std::string> load_image_file(
	const std::filesystem::path& filepath,
	size_t target_width,
	size_t target_height
) {
	const std::string filepath_str = filepath.string();
	if (!std::filesystem::exists(filepath)) {
		return tl::unexpected_fmt("file {} does not exist", filepath_str);
	}
	int width = 0;
	int height = 0;
	int channels = 3;
	float* data =
		stbi_loadf(filepath_str.c_str(), &width, &height, &channels, STBI_rgb);
	if (channels != STBI_rgb) {
		return tl::unexpected_fmt(
			"invalid channels other than RGB in image file {}", filepath_str
		);
	}
	if (data == nullptr) {
		return tl::unexpected_fmt("failed to load image file {}", filepath_str);
	}

	std::vector<float> resized_image(target_width * target_height * STBI_rgb);

	stbir_resize_float_linear(
		data, width, height, 0, resized_image.data(),
		static_cast<int>(target_width), static_cast<int>(target_height), 0,
		STBIR_RGB
	);

	stbi_image_free(data);

	return resized_image;
}

static void linear_to_srgb(std::span<float> values) {
	for (float& linear : values) {
		if (linear <= 0.0031308f)
			linear *= 12.92f;
		else
			linear = 1.055f * powf(linear, 1.0f / 2.4f);
	}
}