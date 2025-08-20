#include "utils.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "datasets/dataset.hpp"
#include <fstream>

tl::expected<std::vector<int8_t>, std::string>
read_binary_file(const std::filesystem::path& filepath) {
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

tl::expected<void, std::string> save_evaluation_result_file(
	const std::filesystem::path& filepath,
	std::span<const float> relative_absolute_pairs
) {
	PROFILE_DEPTH_FUNCTION()

	std::filesystem::create_directories(filepath.parent_path());
	std::ofstream file(filepath);
	if (!file.is_open())
		return tl::unexpected_fmt("Failed to open file: {}", filepath.string());

	file.write(
		reinterpret_cast<const char*>(relative_absolute_pairs.data()),
		static_cast<std::streamsize>(relative_absolute_pairs.size_bytes())
	);

	file.flush();

	return {};
}

static std::string format_span(std::span<const int> s) {
	if (s.empty())
		return "[]";
	std::string result = "[";
	for (size_t i = 0; i < s.size(); ++i) {
		result += std::to_string(s[i]);
		if (i < s.size() - 1)
			result += ", ";
	}
	result += "]";
	return result;
}

tl::expected<EvaluateResult, std::string> evaluate(
	DepthModel& depth_model,
	size_t depth_input_width,
	size_t depth_input_height,
	const RGBDImage& rgbd_image
) {
	PROFILE_DEPTH_FUNCTION()

	size_t pixel_count = rgbd_image.rgb.data().size() / 3;
	if (pixel_count != depth_input_width * depth_input_height) {
		return tl::unexpected_fmt(
			"Invalid image size of {} instead of {}", pixel_count,
			depth_input_width * depth_input_height
		);
	}

	EvaluateResult result;
	result.relative_absolute_pairs.reserve(pixel_count * 2);

	auto image_rgb{rgbd_image.rgb};

	auto depth_estimation_result = depth_model.run_raw(image_rgb);
	if (!depth_estimation_result) {
		return tl::unexpected(depth_estimation_result.error().to_string());
	}
	std::span<float> depth_estimation = depth_estimation_result->data();

	const auto skip_depth_value = [&](size_t image_index) -> bool {
		return rgbd_image.depth_mask && !(*rgbd_image.depth_mask)[image_index];
	};

	for (size_t y = 0; y < depth_input_width; ++y) {
		for (size_t x = 0; x < depth_input_height; ++x) {
			size_t input_image_index = (y * depth_input_width) + x;
			float relative_x =
				static_cast<float>(x) / static_cast<float>(depth_input_width);
			float relative_y =
				static_cast<float>(y) / static_cast<float>(depth_input_height);
			size_t depth_index =
				(static_cast<size_t>(
					 relative_y * static_cast<float>(rgbd_image.depth_height)
				 ) *
				 rgbd_image.depth_width) +
				(static_cast<size_t>(
					relative_x * static_cast<float>(rgbd_image.depth_width)
				));

			if (skip_depth_value(depth_index))
				continue;

			float absolute = rgbd_image.metric_depth[depth_index];

			float relative = depth_estimation[input_image_index];
			result.relative_absolute_pairs.push_back(relative);
			result.relative_absolute_pairs.push_back(absolute);
		}
	}

	return result;
}

tl::expected<FloatTensorBuffer<FloatTensorFormat::ImageRGB>, std::string>
load_rgb_image_file(
	const std::filesystem::path& filepath,
	size_t target_width,
	size_t target_height
) {
	PROFILE_DEPTH_FUNCTION()

	const std::string filepath_str = filepath.string();
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

	return FloatTensorBuffer<FloatTensorFormat::ImageRGB>(
		std::move(resized_image)
	);
}

FloatTensorBuffer<FloatTensorFormat::ImageRGB255>
image_rgb_255_operator(FloatTensorBuffer<FloatTensorFormat::ImageRGB>& input) {
	auto values = input.data();

	for (float& value : values) {
		value = std::clamp(value * 255.f, 0.f, 255.f);
	}

	return input.convert_format<FloatTensorFormat::ImageRGB255>();
}

tl::expected<std::vector<uint16_t>, std::string>
load_16bit_greyscale_image_file(
	const std::filesystem::path& filepath,
	size_t& out_width,
	size_t& out_height
) {
	PROFILE_DEPTH_FUNCTION()

	const std::string filepath_str = filepath.string();
	int width = 0;
	int height = 0;
	int channels = STBI_grey;
	stbi_us* data = stbi_load_16(
		filepath_str.c_str(), &width, &height, &channels, STBI_grey
	);
	if (channels != STBI_grey) {
		return tl::unexpected_fmt(
			"invalid channels other than greyscale in image file {}",
			filepath_str
		);
	}
	if (data == nullptr) {
		return tl::unexpected_fmt(
			"failed to load greyscale image file {}", filepath_str
		);
	}

	std::vector<uint16_t> image(static_cast<size_t>(width * height));
	out_width = static_cast<size_t>(width);
	out_height = static_cast<size_t>(height);
	std::memcpy(
		image.data(), data, image.size() * sizeof(decltype(image)::value_type)
	);

	stbi_image_free(data);

	return image;
}

tl::expected<std::vector<float>, std::string>
load_npy_file(const std::filesystem::path& filepath) {
	PROFILE_DEPTH_FUNCTION()

	// first try loading as float
	try {
		const auto npy_data = npy::read_npy<float>(filepath);
		return npy_data.data;
	} catch (const std::exception& e) {
		// then as double -> float
		try {
			const auto npy_data = npy::read_npy<double>(filepath);
			std::vector<float> values(npy_data.data.size());
			for (size_t i = 0; i < values.size(); ++i)
				values[i] = static_cast<float>(npy_data.data[i]);
			return values;
		} catch (const std::exception& e) {
			return tl::unexpected_fmt(
				"failed to load npy file {}: {}", filepath.string(), e.what()
			);
		}
	}
}

tl::expected<std::chrono::milliseconds, std::string> evaluate_datapoint(
	DepthModel& depth_model,
	const RGBDDataPoint& datapoint,
	const std::filesystem::path& evaluation_output_filepath
) {
	PROFILE_DEPTH_FUNCTION()

	const auto start = std::chrono::high_resolution_clock::now();

	const std::span<const int> input_shape = depth_model.get_input_shape();
	if (input_shape.size() != 4) {
		return tl::unexpected_fmt(
			"invalid input shape dimensions, expected 4 but has {}",
			input_shape.size()
		);
	}
	if (input_shape[0] != 1) {
		return tl::unexpected_fmt(
			"invalid batch size, expected 1 but has {}", input_shape[0]
		);
	}
	if (input_shape[3] != 3) {
		return tl::unexpected_fmt(
			"invalid channel size, expected 3 (r,g,b) but has {}",
			input_shape[3]
		);
	}
	auto depth_input_width = static_cast<size_t>(input_shape[2]);
	auto depth_input_height = static_cast<size_t>(input_shape[1]);

	const std::span<const int> output_shape = depth_model.get_output_shape();
	const std::array<int, 4> expected_output_shape{
		1, input_shape[1], input_shape[2], 1
	};
	if (!std::ranges::equal(output_shape, expected_output_shape)) {
		return tl::unexpected_fmt(
			"invalid output shape, expected {} but has {}",
			format_span(expected_output_shape), format_span(output_shape)
		);
	}

	auto rgbd_image_result =
		datapoint.load(depth_input_width, depth_input_height);
	if (!rgbd_image_result)
		return tl::unexpected(rgbd_image_result.error());
	RGBDImage& rgbd_image = rgbd_image_result.value();

	if (rgbd_image.rgb.data().size() !=
		depth_input_width * depth_input_height * 3) {
		return tl::unexpected_fmt(
			"invalid image size of {} pixels, expected {}x{}={} pixels",
			rgbd_image.rgb.data().size() / 3, depth_input_width,
			depth_input_height, depth_input_width * depth_input_height * 3
		);
	}

	const auto result = evaluate(
		depth_model, depth_input_width, depth_input_height, rgbd_image
	);
	if (!result.has_value())
		return tl::unexpected(result.error());

	const auto save_result = save_evaluation_result_file(
		evaluation_output_filepath,
		std::span<const float>(result.value().relative_absolute_pairs)
	);
	if (!save_result.has_value())
		return tl::unexpected(save_result.error());

	return std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::high_resolution_clock::now() - start
	);
}