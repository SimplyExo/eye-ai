#include "EyeAICore/audio/SpacialAudio.hpp"
#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

SpacialAudio::SpacialAudio() {}

void SpacialAudio::getDepthEstimationData(std::vector<float> data) {
	this->depthEstimationData = data;
}

void SpacialAudio::processDepthEstimationData() {

	coloum_length = depthEstimationData.size() / row_length;
	for (int i = 0; i < row_length; ++i) {
		// extract the colounm out of the data
		std::vector<float> coloum;
		for (int j = 0; j < coloum_length; ++j) {
			coloum.push_back(depthEstimationData.at(i + ( j * row_length)));
		}

		float nearest_distance =
			*std::min_element(coloum.begin(), coloum.end());

		std::array<float, 3> sound_origin =
			CalculateSoundOrigin().calculateSoundOrigin(
				std::array<int, 2>{i + 1 , 0}, nearest_distance
			);
		std::cout << "i: " << i << "\n";
		std::cout << "x1: " <<sound_origin[0] << "\n";
		std::cout << "x2: " <<sound_origin[1] << "\n";
		std::cout << "distance: " << nearest_distance << "\n";
		audio_main.playSound(200.0f, 1.0f, sound_origin);
	}
}

SpacialAudio::~SpacialAudio() {}