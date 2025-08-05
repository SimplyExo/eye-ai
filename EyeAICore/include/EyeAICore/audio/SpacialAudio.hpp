#pragma once
#include "EyeAICore/audio/AudioMain.hpp"
#include <AL/al.h>
#include <AL/alc.h>
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
	int row_length = 4;
	int coloum_length;
	AudioMain audio_main;

  public:
	SpacialAudio();
	~SpacialAudio();
	void getDepthEstimationData(std::vector<float> data);
	void processDepthEstimationData();
};
