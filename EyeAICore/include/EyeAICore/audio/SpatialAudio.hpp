#pragma once

/*
SpatialAudio handles everything necessary for the spatial audio:
- retreving the depthEstimationData
- converting depthEstimationData into audio source position
- managing AudioMain instance
*/

#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include "EyeAICore/YoloModel.hpp"

#include <AL/al.h>
#include <AL/alc.h>
#include <string>
#include <thread>
#include <vector>
#include <span>
#include <array>
#include <unordered_map>


class SpatialAudio {
  private:
	std::array<float, 256 * 256> depthEstimationData = {0};
	std::vector<YoloModel::BoundingBox> objectDetectionData;
	int row_length = 256;
	int column_length = 256;
	bool isFinished = true;
	AudioMain audio_main;
	const SpatialAudioSettings& audio_settings;
	void readObjectLabelData();
	std::unordered_map<std::string, std::array<int, 2>> object_label_data;

  public:
	SpatialAudio(const SpatialAudioSettings& audio_settings);
	~SpatialAudio();
	void getAIData(std::span<float, 256 * 256> depth_estimation_data, std::vector<YoloModel::BoundingBox> object_detection_data);
	void processDepthEstimationData();
	void processObjectDetectionData();
	bool getProcessingStatus();
	

	std::thread depth_audio_thread;
	std::atomic<bool> depth_audio_running{true};
	std::thread object_audio_thread;
	std::atomic<bool> object_audio_running{true};
};
