#pragma once

/*
SpatialAudio handles everything necessary for the spatial audio:
- retreving the depthEstimationData
- converting depthEstimationData into audio source position
- managing AudioMain instance
*/

#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/AudioSettings.hpp"

#include <AL/al.h>
#include <AL/alc.h>
#include <thread>
#include <vector>
#include <span>
#include <array>


class SpatialAudio {
  private:
	std::array<float, 256 * 256> depthEstimationData = {0};
	int row_length = 256;
	int column_length = 256;
	bool isFinished = true;
	AudioMain audio_main;
	const AudioSettings& audio_settings;

  public:
	SpatialAudio(const AudioSettings& audio_settings);
	~SpatialAudio();
	void getDepthEstimationData(std::span<float, 256 * 256> data);
	void processDepthEstimationData();
	bool getProcessingStatus();

	std::thread audio_thread;
	std::atomic<bool> running{true};
};
