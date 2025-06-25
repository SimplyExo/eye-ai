#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>

constexpr const char* GPU_DELEGATE_SERIALIZATION_DIR = "/tmp/EyeAICore/gpu_delegate_cache";
constexpr const char* MIDAS_MODEL_TOKEN = "ijustmadethistokenup";
constexpr size_t INPUT_WIDTH = 640;
constexpr size_t INPUT_HEIGHT = 640;
constexpr size_t OUTPUT_BUFFER_SIZE = 84 * 8400;
constexpr std::array<float, 3> MEAN = {123.675f, 116.28f, 103.53f};
constexpr std::array<float, 3> STDDEV = {58.395f, 57.12f, 57.375f};

template<typename T>
static tl::expected<std::vector<T>, std::string>
read_binary_file(const std::filesystem::path& filepath);


void print_vector(std::vector<float> v)
{
    for (int i = 0; i < v.size(); i++)
    {
        std::cout << v[i] << std::endl;
    }
}

int main() {
    std::filesystem::create_directories(GPU_DELEGATE_SERIALIZATION_DIR);

    auto model_data_result = read_binary_file<int8_t>("/home/thomas/StudioProjects/eye-ai/EyeAIApp/app/src/main/assets/yolo11n_float32.tflite");
    if (!model_data_result.has_value()) {
        std::cout << "Failed to read model file: " << model_data_result.error();
        return 1;
    }
    auto& model_data = model_data_result.value();

    TfLiteLogWarningCallback tflite_log_warning_callback = [](std::string msg) {
        std::cout << "[TfLite Warning] " << msg;
    };
    TfLiteLogErrorCallback tflite_log_error_callback = [](std::string msg) {
        std::cout << "[TfLite Error] " << msg;
    };

    YoloModel yolo_instance;

    auto result = yolo_instance.create(
        std::move(model_data),
        GPU_DELEGATE_SERIALIZATION_DIR,
        MIDAS_MODEL_TOKEN,
        tflite_log_warning_callback,
        tflite_log_error_callback
    );

    if (!result.has_value()) {
        std::cout << "Could not create YoloModel: " << result.error();
        return 1;
    }

    std::cout << "Runtime erstellt!\n";

    std::vector<float> input_image(INPUT_WIDTH * INPUT_HEIGHT * 3);
    std::vector<float> object_recognition_output(OUTPUT_BUFFER_SIZE);


    const auto exec = yolo_instance.run(input_image, object_recognition_output);

    if (!exec.has_value()) {
        std::cout << "Failed to run depth estimation: " << exec.error();
    }

    else {
        std::cout << "Success running calculation!\n";
    }

    print_vector(object_recognition_output);

    return 0;
}

template<typename T>
static tl::expected<std::vector<T>, std::string>
read_binary_file(const std::filesystem::path& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);

    if (!file.is_open())
        return tl::unexpected_fmt("Failed to open file: {}", filepath.string());

    std::streamsize binary_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (binary_size % sizeof(T) != 0) {
        return tl::unexpected_fmt(
            "File size {} is not a multiple of sizeof({})",
            binary_size,
            typeid(T).name()
        );
    }

    std::vector<T> buffer(binary_size / sizeof(T));

    if (!file.read(reinterpret_cast<char*>(buffer.data()), binary_size))
        return tl::unexpected_fmt("Failed to read file: {}", filepath.string());

    return buffer;
}
