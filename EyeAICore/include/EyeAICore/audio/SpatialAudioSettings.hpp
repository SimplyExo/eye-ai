#pragma once

#include <cstddef>
#include <sndfile.hh>
#include <string>
#include <vector>

/*
All the settings or information which needs to
be accessible by multiple classes is stored here.
Can partially be changed by the EyeAIApp to give
the user customization options
*/

using SpatialAudioLogErrorCallback = void (*)(std::string);
using SpatialAudioLogInfoCallback = void (*)(std::string);

class SpatialAudioSettings {
  public:
	// constants
	constexpr static int BUFFERS_PER_SOURCE = 3;
	constexpr static int SAMPLE_RATE = 48000;
	constexpr static int picture_x_resolution = 256;
	constexpr static int picture_y_resolution = 256;
	constexpr static int NUMBER_OF_SOURCES = 9;
	// files containing the data and audio of the objects
	std::vector<std::byte> coco_labels_audio;
	std::vector<std::byte> coco_labels_data;
	// pausing the playback
	bool depth_audio_paused = false;
	bool object_audio_paused = false;
	float FREQUENCY;
	float BUFFER_DURATION = 1.0f;

	SpatialAudioLogErrorCallback logErrorCallback;
	SpatialAudioLogInfoCallback logInfoCallback;

	SpatialAudioSettings(
		SpatialAudioLogErrorCallback logErrorCallback,
		SpatialAudioLogErrorCallback logInfoCallback,
		float freq = 500.0f,
		float buf_dur = 0.25
	)
		: FREQUENCY(freq),BUFFER_DURATION(buf_dur), 
		  logErrorCallback(logErrorCallback), logInfoCallback(logInfoCallback) {
	}
};