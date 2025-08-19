#include "utils.hpp"

tl::expected<std::vector<int8_t>, std::string>
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

tl::expected<std::unique_ptr<DepthModel>, std::string>
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

tl::expected<std::unique_ptr<TfLiteRuntime>, std::string>
create_test_tflite_runtime(
	const std::filesystem::path& model_path,
	FloatTensorFormat model_input_format,
	FloatTensorFormat model_output_format,
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
		[](const std::string msg) { std::cout << "[WARN]  " << msg << '\n'; },
		[](const std::string msg) { std::cerr << "[ERROR] " << msg << '\n'; },
		profiling_frame
	);
	if (runtime_result)
		return std::move(runtime_result.value());
	return tl::unexpected(runtime_result.error().to_string());
}

tl::expected<FloatTensorBuffer<FloatTensorFormat::ImageRGB>, std::string>
load_image_file(
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
		data_float[i] = static_cast<float>(data[i]) / 255.f;
		data_float[i] = std::clamp(data_float[i], 0.f, 255.f);
	}

	std::vector<float> resized_image(target_width * target_height * STBI_rgb);

	stbir_resize_float_linear(
		data_float.data(), width, height, 0, resized_image.data(),
		static_cast<int>(target_width), static_cast<int>(target_height), 0,
		STBIR_RGB
	);

	stbi_image_free(data_ptr);

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

tl::expected<std::vector<std::string>, std::string>
read_coco_labels_file(const std::filesystem::path& filepath) {
	std::vector<std::string> labels;
	std::ifstream file(filepath);
	if (!file.is_open()) {
		return tl::unexpected(
			std::format("Failed to open file: {}", filepath.string())
		);
	}

	std::string line;
	while (std::getline(file, line)) {
		labels.push_back(line);
	}
	file.close();
	return labels;
}