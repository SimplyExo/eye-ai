#include "EyeAICore/audio/SpacialAudio.hpp"
#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/AudioSourceData.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include <algorithm>
#include <thread>
#include <vector>

SpacialAudio::SpacialAudio() {
	audio_thread =
		std::thread([this]() { audio_main.startAudioLoop(running); });
}

void SpacialAudio::getDepthEstimationData(std::vector<float> data) {
	this->depthEstimationData = data;
	processDepthEstimationData();
}

void SpacialAudio::processDepthEstimationData() {
	/*
	Processes the depth-estimation data:
	- dividing data into columns
	- getting nearest distance in column
	- calculate sound origin of nearest distance
	- saving this in AudioSource
	- updating audio sources in AudioMain
	Note: depending on the value of NUMBER_OF_SOURCES,
	not always all columns are used
	*/

	std::vector<AudioSourceData> new_audio_source_data;
	isFinished = false;
	column_length = depthEstimationData.size() / row_length;
	int step_size = row_length / NUMBER_OF_SOURCES; // NUMBER_OF_SOURCES = 2^x!

	for (int i = 0; i < row_length; i += step_size) {
		/*
		Because the data doesn't come in a 2d form, it is
		necessary to extract all elements of a column first.
		i + (j * row_length) represents all elements of a column.
		*/
		std::vector<float> column;
		for (int j = 0; j < column_length; ++j) {
			column.push_back(depthEstimationData.at(i + (j * row_length)));
		}

		float nearest_distance =
			*std::min_element(column.begin(), column.end()) *
			10; // TODO: die *10 sind nur für Testzwecke

		/*
		Retrieving sound origin, by passing coordinates in Frame and nearest
		distance to the CalculateSoundOrigin::calculateSoundOrigin()
		function, which retruns the origin as a vector of coordinates
		*/
		std::array<float, 3> sound_origin =
			CalculateSoundOrigin().calculateSoundOrigin(
				std::array<int, 2>{i + 1, 0}, nearest_distance
			);

		new_audio_source_data.push_back(AudioSourceData{
			200.0f, BUFFER_LENGTH,sound_origin[0], sound_origin[1], sound_origin[2]
		});
	}
	audio_main.changeAudioData(new_audio_source_data);
	isFinished = true;
}

bool SpacialAudio::getProcessingStatus() { return isFinished; }

SpacialAudio::~SpacialAudio() {
	running = false;
	if (audio_thread.joinable()) {
		audio_thread.join();
	}
}