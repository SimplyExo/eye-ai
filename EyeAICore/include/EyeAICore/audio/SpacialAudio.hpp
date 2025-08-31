#pragma once

/*
SpacialAudio handles everything necessary for the spacial audio:
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

class SpacialAudio {
  private:
	std::array<float, 256 * 256> depthEstimationData = {0};
	int row_length = 265;
	int column_length = 265;
	bool isFinished = true;
	AudioMain audio_main;
	const int NUMBER_OF_SOURCES = 16;
	const float BUFFER_LENGTH = 1;
	//const int SAMPLE_RATE = 44100;

  public:
	SpacialAudio();
	~SpacialAudio();
	void getDepthEstimationData(std::span<float, 256 * 256> data);
	void processDepthEstimationData();
	bool getProcessingStatus();

	std::thread audio_thread;
	std::atomic<bool> running{true};
};
