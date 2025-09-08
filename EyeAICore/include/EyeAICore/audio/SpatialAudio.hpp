#pragma once

#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <array>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

/*
SpatialAudio handles everything necessary for the spatial audio:
- retreving the AI-Data
- converting depthEstimationData into audio source positions
- converting objectDetectionData into audio source positions
- managing AudioMain instance
*/

class SpatialAudio {
  public:
	SpatialAudio(const SpatialAudioSettings& audio_settings);
	~SpatialAudio();

	void getAIData(
		std::span<float, 256 * 256> depth_estimation_data,
		std::vector<YoloModel::BoundingBox> object_detection_data
	);
	void processDepthEstimationData();
	void processObjectDetectionData();
	bool getProcessingStatus();

	// thread handling
	std::thread depth_audio_thread;
	std::atomic<bool> depth_audio_running{true};
	std::thread object_audio_thread;
	std::atomic<bool> object_audio_running{true};

  private:
	// the ai data
	std::array<float, 256 * 256> depthEstimationData = {0};
	std::vector<YoloModel::BoundingBox> objectDetectionData;

	bool processingFinished = true;
	AudioMain audio_main;

	// global settings for spatial audio
	const SpatialAudioSettings& audio_settings;

	// handling the data of the objects
	void readObjectLabelData();
	std::unordered_map<std::string, std::array<int, 2>> object_label_data;
};
