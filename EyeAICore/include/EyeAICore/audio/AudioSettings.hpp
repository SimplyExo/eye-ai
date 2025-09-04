#pragma once

#include <string>

using SpatialAudioLogErrorCallback = void (*)(std::string);
using SpatialAudioLogInfoCallback = void (*)(std::string);


class AudioSettings {
  public:
	constexpr static float BUFFER_DURATION = 1.0f;
	constexpr static int BUFFERS_PER_SOURCE = 3;
	constexpr static int SAMPLE_RATE = 48000;
	int NUMBER_OF_SOURCES;
	float FREQUENCY;

	SpatialAudioLogErrorCallback logErrorCallback;
	SpatialAudioLogInfoCallback logInfoCallback;

	AudioSettings(
		SpatialAudioLogErrorCallback logErrorCallback,
		SpatialAudioLogErrorCallback logInfoCallback,
		int num_of_sources = 8,
		float freq = 150.0f
	)
		: NUMBER_OF_SOURCES(num_of_sources), FREQUENCY(freq),
		  logErrorCallback(logErrorCallback), logInfoCallback(logInfoCallback) {
	}
};