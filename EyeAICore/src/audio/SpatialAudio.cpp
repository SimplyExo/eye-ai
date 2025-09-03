#include "EyeAICore/audio/SpatialAudio.hpp"
#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/AudioSourceData.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include <algorithm>
#include <thread>
#include <vector>
#include <iostream>
#include <span>
#include <cmath>

static SpatialAudioLogErrorCallback logErrorCallback = nullptr;
static SpatialAudioLogInfoCallback logInfoCallback = nullptr;

void setSpatialAudioLogErrorCallback(SpatialAudioLogErrorCallback callback){
	logErrorCallback = callback;
}

void setSpatialAudioLogInfoCallback(SpatialAudioLogInfoCallback callback){
	logInfoCallback = callback;
}

SpatialAudio::SpatialAudio() {
	audio_thread =
		std::thread([this]() { audio_main.startAudioLoop(running); });
}

void SpatialAudio::getDepthEstimationData(std::span<float, 256 * 256> data) {
	std::ranges::copy(data, this->depthEstimationData.begin());
	processDepthEstimationData();
}

void SpatialAudio::processDepthEstimationData() {
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
	int step_size = row_length / audio_settings.NUMBER_OF_SOURCES; // NUMBER_OF_SOURCES = 2^x!

	//LOG_INFO("[ProcessDepthEstimationData] Started processing...");
	logErrorCallback("[ProcessDepthEstimationData] Started processing...");

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
			*std::min_element(column.begin(), column.end()); 

		//LOG_INFO("[ProcessDepthEstimationData] Nearest Distance for Source {}: {}", i, nearest_distance);
		/*
		Retrieving sound origin, by passing coordinates in Frame and nearest
		distance to the CalculateSoundOrigin::calculateSoundOrigin()
		function, which retruns the origin as a vector of coordinates
		*/
		std::array<float, 3> sound_origin =
			CalculateSoundOrigin().calculateSoundOrigin(
				std::array<int, 2>{i + 1, 0}, nearest_distance
			);
			//float actual_distance = sqrt(sound_origin[0] * sound_origin[0] * sound_origin[1] * sound_origin[1]);

		new_audio_source_data.push_back(AudioSourceData{
			200.0f, audio_settings.BUFFER_DURATION,audio_settings.SAMPLE_RATE,sound_origin[0], sound_origin[1], sound_origin[2]
		});
	}
	audio_main.changeAudioData(new_audio_source_data);
	isFinished = true;
	//LOG_INFO("[ProcessDepthEstimationData] Finished processing...");
}

bool SpatialAudio::getProcessingStatus() { return isFinished; }

SpatialAudio::~SpatialAudio() {
	running = false;
	if (audio_thread.joinable()) {
		audio_thread.join();
	}
}