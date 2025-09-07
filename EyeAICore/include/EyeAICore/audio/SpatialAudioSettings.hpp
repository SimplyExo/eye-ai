#pragma once

#include <cstddef>
#include <string>
#include <sndfile.hh>
#include <vector>

using SpatialAudioLogErrorCallback = void (*)(std::string);
using SpatialAudioLogInfoCallback = void (*)(std::string);


class SpatialAudioSettings {
  public:
	constexpr static float BUFFER_DURATION = 1.0f;
	constexpr static int BUFFERS_PER_SOURCE = 3;
	constexpr static int SAMPLE_RATE = 48000;
	constexpr static int picture_x_resolution = 256;
	constexpr static int picture_y_resolution = 256;
	std::vector<std::byte> coco_labels_audio;
	std::vector<std::byte> coco_labels_data;
	int NUMBER_OF_SOURCES;
	float FREQUENCY;

	SpatialAudioLogErrorCallback logErrorCallback;
	SpatialAudioLogInfoCallback logInfoCallback;

	SpatialAudioSettings(
		SpatialAudioLogErrorCallback logErrorCallback,
		SpatialAudioLogErrorCallback logInfoCallback,
		int num_of_sources = 8,
		float freq = 150.0f
	)
		: NUMBER_OF_SOURCES(num_of_sources), FREQUENCY(freq),
		  logErrorCallback(logErrorCallback), logInfoCallback(logInfoCallback) {
	}
};