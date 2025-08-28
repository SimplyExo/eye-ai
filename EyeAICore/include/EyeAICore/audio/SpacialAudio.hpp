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

class SpacialAudio {
  private:
	std::vector<float> depthEstimationData = {
		1, //0
		0.5, //1
		1.23, //2
		2.0, //3
		0.75, //0
		3.4, //1
		2.3, //2
		1.89, //3
		0.78, //0
		1.23, //1
		2.34, //2
		0.93, //3
	};
	int row_length = 265;
	int column_length = 265;
	bool isFinished = true;
	AudioMain audio_main;
	const int NUMBER_OF_SOURCES = 16;
	const float BUFFER_LENGTH = 1;
	const int SAMPLE_RATE = 44100;
	

  public:
	SpacialAudio();
	~SpacialAudio();
	void getDepthEstimationData(std::vector<float> data);
	void processDepthEstimationData();
	bool getProcessingStatus();

	std::thread audio_thread;
	std::atomic<bool> running{true};

};
