#pragma once

/*
SpatialAudio handles everything necessary for the spatial audio:
- retreving the depthEstimationData
- converting depthEstimationData into audio source position
- managing AudioMain instance
*/

#include "EyeAICore/audio/AudioMain.hpp"

#include <AL/al.h>
#include <AL/alc.h>
#include <thread>
#include <vector>
#include <span>
#include <array>

using SpatialAudioLogErrorCallback = void (*)(std::string);
using SpatialAudioLogInfoCallback = void (*)(std::string);


class SpatialAudio {
  private:
	std::array<float, 256 * 256> depthEstimationData = {0};
	int row_length = 256;
	int column_length = 256;
	bool isFinished = true;
	AudioMain audio_main;
	AudioSettings audio_settings;

  public:
	SpatialAudio();
	~SpatialAudio();
	void getDepthEstimationData(std::span<float, 256 * 256> data);
	void processDepthEstimationData();
	bool getProcessingStatus();

	std::thread audio_thread;
	std::atomic<bool> running{true};

	void setSpatialAudioLogErrorCallback(SpatialAudioLogErrorCallback callback);
	void setSpatialAudioLogInfoCallback(SpatialAudioLogErrorCallback callback);
};
