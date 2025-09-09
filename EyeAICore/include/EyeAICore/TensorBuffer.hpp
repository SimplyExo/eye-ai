#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <variant>
#include <vector>

/// non owning buffer of T, lifetime of underlying std::span<T> should be
/// managed externally
template<typename T, typename Format, Format FORMAT_CONSTANT>
struct TensorBuffer {
	explicit TensorBuffer(
		std::variant<std::shared_ptr<std::vector<T>>, std::span<T>> data
	)
		: data_container(std::move(data)) {}
	explicit TensorBuffer(std::span<T> data) : data_container(data) {}
	explicit TensorBuffer(std::shared_ptr<std::vector<T>> data)
		: data_container(std::move(data)) {}
	explicit TensorBuffer(std::vector<T>&& data)
		: data_container(std::make_shared<std::vector<T>>(std::move(data))) {}

	TensorBuffer(const TensorBuffer&) = default;
	TensorBuffer(TensorBuffer&&) = default;
	TensorBuffer& operator=(const TensorBuffer&) = default;
	TensorBuffer& operator=(TensorBuffer&&) = default;
	~TensorBuffer() = default;

	template<Format NEW_FORMAT>
	[[nodiscard]] TensorBuffer<T, Format, NEW_FORMAT> convert_format() const {
		return TensorBuffer<T, Format, NEW_FORMAT>(data_container);
	}

	[[nodiscard]] std::span<T> data() {
		return std::visit(
			[](auto& data_container) -> std::span<T> {
				using data_container_t = std::decay_t<decltype(data_container)>;

				if constexpr (std::is_same_v<
								  data_container_t,
								  std::shared_ptr<std::vector<T>>>) {
					return std::span<T>(*data_container);
				} else if constexpr (std::is_same_v<
										 data_container_t, std::span<T>>) {
					return data_container;
				}
			},
			data_container
		);
	}
	[[nodiscard]] std::span<const T> data() const {
		return std::visit(
			[](const auto& data_container) -> std::span<const T> {
				using data_container_t = std::decay_t<decltype(data_container)>;

				if constexpr (std::is_same_v<
								  data_container_t,
								  std::shared_ptr<std::vector<T>>>) {
					return std::span<const T>(*data_container);
				} else if constexpr (std::is_same_v<
										 data_container_t, std::span<T>>) {
					return data_container;
				}
			},
			data_container
		);
	}

  private:
	std::variant<std::shared_ptr<std::vector<T>>, std::span<T>> data_container;
};

enum class FloatTensorFormat : std::uint8_t {
	/// 3 floats for r, g, b, all in range [0.f, 1.f]
	ImageRGB,
	/// 3 floats for r, g, b, all in range [0.f, 255.f]
	ImageRGB255,
	/// 3 floats for r, g, b, each in the range [-1.f, 1.f]
	MiDaSImageRGB,
	/// 3 floats for r, g, b, all in range [0.f, 1.f]
	YoloImageRGB,
	/// 1 float per pixel for relative depth in range [0.f, 1.f]
	RelativeDepth,
	/// 1 float per pixel for relative depth, all values possible
	RawRelativeDepth,
	/// 4 floats per pixel (RGBD) where RGB is in range [-1.f, 1.f] (colorspace:
	/// sRGB) and D is the raw relative depth remapped from [0.f, 1500.f] to
	/// [-1.f, 1.f] (and clamped!)
	Rel2AbsDepthInput,
	/// simple float array of the coefficients of the polynomial relative to
	/// absolute function produced by the RelToAbs model
	Rel2AbsDepthCoefficientOutput,
	/// 1 float per pixel for metric depth in meters
	MetricDepth,
	/// special yolo output of detected objects and their confidence
	YoloOutput
};

std::string_view format_float_tensor_format(FloatTensorFormat format);

template<FloatTensorFormat FORMAT>
using FloatTensorBuffer = TensorBuffer<float, FloatTensorFormat, FORMAT>;