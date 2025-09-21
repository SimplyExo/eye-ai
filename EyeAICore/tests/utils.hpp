#pragma once

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/MetricDepthModel.hpp"
#include "EyeAICore/TensorBuffer.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include <filesystem>
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

tl::expected<std::unique_ptr<MetricDepthModel>, std::string>
create_test_metric_depth_model();

tl::expected<FloatTensorBuffer<FloatTensorFormat::ImageRGB>, std::string>
load_image_file(
	const std::filesystem::path& filepath,
	size_t target_width,
	size_t target_height
);

FloatTensorBuffer<FloatTensorFormat::ImageRGB255>
image_rgb_255_operator(FloatTensorBuffer<FloatTensorFormat::ImageRGB>& input);

tl::expected<std::vector<std::string>, std::string>
read_coco_labels_file(const std::filesystem::path& filepath);