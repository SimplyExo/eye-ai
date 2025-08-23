#include "utils.hpp"
#include "EyeAICore/Rel2AbsDepthModel.hpp"
#include "EyeAICore/TensorBuffer.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "datasets/dataset.hpp"
#include <Eigen/Dense>
#include <algorithm>
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

tl::expected<PreparedImage, std::string> prepare(
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

	/// [rel0, abs0, rel1, abs1, ...]
	std::vector<float> relative_absolute_pairs;
	relative_absolute_pairs.reserve(pixel_count * 2);

	/// create deep copy of rgbd_image.rgb, image_rgb will be modified!
	auto image_rgb_span = rgbd_image.rgb.data();
	auto image_rgb = FloatTensorBuffer<FloatTensorFormat::ImageRGB255>{
		std::vector<float>(image_rgb_span.begin(), image_rgb_span.end())
	};

	auto depth_estimation_result = depth_model.run_raw(image_rgb);
	if (!depth_estimation_result) {
		return tl::unexpected(depth_estimation_result.error().to_string());
	}
	auto& depth_estimation = *depth_estimation_result;
	std::span<float> depth_estimation_values = depth_estimation.data();

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

			float relative = depth_estimation_values[input_image_index];
			relative_absolute_pairs.push_back(relative);
			relative_absolute_pairs.push_back(absolute);
		}
	}

	auto coeffs = find_coeffs(relative_absolute_pairs);
	auto rgbd = rel2abs_input_operator(rgbd_image.rgb, depth_estimation);

	return PreparedImage(rgbd, coeffs);
}

tl::expected<std::chrono::milliseconds, std::string> prepare_datapoint(
	DepthModel& depth_model,
	const RGBDDataPoint& datapoint,
	const std::filesystem::path& prepared_output_directory,
	size_t datapoint_index
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

	const auto result =
		prepare(depth_model, depth_input_width, depth_input_height, rgbd_image);
	if (!result.has_value())
		return tl::unexpected(result.error());
	const auto& prepared_rgbd_image = result.value();

	try {
		const auto rgbd_output_filepath =
			prepared_output_directory /
			std::format("{}_rgbd.npy", datapoint_index);
		const auto rgbd_data = prepared_rgbd_image.rgbd.data();
		assert(rgbd_data.size() == depth_input_width * depth_input_height * 4);
		npy::write_npy(
			rgbd_output_filepath,
			npy::npy_data_ptr<float>{
				.data_ptr = rgbd_data.data(),
				.shape = {depth_input_width, depth_input_height, 4}
			}
		);
		const auto coeffs_output_filepath =
			prepared_output_directory /
			std::format("{}_coeffs.npy", datapoint_index);
		const auto coeffs_data = prepared_rgbd_image.coeffs.data();
		assert(coeffs_data.size() == Rel2AbsDepthModel::COEFFS_COUNT);
		npy::write_npy(
			coeffs_output_filepath,
			npy::npy_data_ptr<float>{
				.data_ptr = coeffs_data.data(), .shape = {coeffs_data.size()}
			}
		);
	} catch (const std::exception& e) {
		return tl::unexpected_fmt(
			"failed to write prepared files for datapoint {}:\n{}",
			datapoint_index, e.what()
		);
	}

	return std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::high_resolution_clock::now() - start
	);
}

/// does the same as np.polyfit
FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthCoefficientOutput>
find_coeffs(std::span<const float> relative_absolute_pairs) {
	long n = (long)relative_absolute_pairs.size() / 2;
	Eigen::MatrixXf X(n, Rel2AbsDepthModel::COEFFS_COUNT);
	Eigen::VectorXf Y(n);

	// Build Vandermonde matrix
	for (long i = 0; i < n; i++) {
		float xi = 1.0;
		for (long j = 0; j <= Rel2AbsDepthModel::POLYNOMIAL_DEGREE; j++) {
			X(i, j) = xi;
			float x = relative_absolute_pairs[i * 2];
			xi *= x;
		}
		float y = relative_absolute_pairs[(i * 2) + 1];
		Y(i) = y;
	}

	// Solve normal equations: (X^T X) c = X^T y
	Eigen::VectorXf coeffs =
		(X.transpose() * X).ldlt().solve(X.transpose() * Y);
	std::span<float> coeffs_span(coeffs.data(), coeffs.size());

	return FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthCoefficientOutput>{
		std::vector<float>(coeffs_span.begin(), coeffs_span.end())
	};
}

tl::expected<FloatTensorBuffer<FloatTensorFormat::ImageRGB255>, std::string>
load_rgb_image_file(
	const std::filesystem::path& filepath,
	size_t target_width,
	size_t target_height
) {
	PROFILE_DEPTH_FUNCTION()

	const std::string filepath_str = filepath.string();
	if (!std::filesystem::exists(filepath)) {
		return tl::unexpected_fmt("file {} does not exist", filepath_str);
	}
	int width = 0;
	int height = 0;
	int channels = 3;
	stbi_uc* data_ptr =
		stbi_load(filepath_str.c_str(), &width, &height, &channels, STBI_rgb);
	if (channels != STBI_rgb) {
		return tl::unexpected_fmt(
			"invalid channels other than RGB in image file {}", filepath_str
		);
	}
	if (data_ptr == nullptr) {
		return tl::unexpected_fmt("failed to load image file {}", filepath_str);
	}

	std::span<stbi_uc> data(
		data_ptr, static_cast<size_t>(width * height * channels)
	);
	std::vector<float> data_float(data.size());
	for (size_t i = 0; i < data_float.size(); ++i) {
		data_float[i] = static_cast<float>(data[i]);
		data_float[i] = std::clamp(data_float[i], 0.f, 255.f);
	}

	std::vector<float> resized_image(target_width * target_height * STBI_rgb);

	stbir_resize_float_linear(
		data_float.data(), width, height, 0, resized_image.data(),
		static_cast<int>(target_width), static_cast<int>(target_height), 0,
		STBIR_RGB
	);

	stbi_image_free(data_ptr);

	return FloatTensorBuffer<FloatTensorFormat::ImageRGB255>(
		std::move(resized_image)
	);
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