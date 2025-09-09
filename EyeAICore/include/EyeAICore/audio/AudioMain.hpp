#pragma once

#include "EyeAICore/audio/DepthAudioSourceData.hpp"
#include "EyeAICore/audio/ObjectAudioSourceData.hpp"
#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <atomic>
#include <vector>

/*
AudioMain handles the audio playback:
- managing the playback of the
  depth estimation data
- managing the playback of the objects
  recognized by the object detection
Once these loops start, continues playback
is ensured
*/

class AudioMain {
  public:
	AudioMain(const SpatialAudioSettings& audio_settings);
	~AudioMain();

	// Functions for playing depth estimation data
	void startDepthAudioLoop(std::atomic<bool>& running);
	void changeDepthAudioData(
		std::vector<DepthAudioSourceData> new_audio_source_data
	);

	// Functions for playing object detection data
	void startObjectAudioLoop(std::atomic<bool>& running);
	void changeObjectAudioData(
		std::vector<ObjectAudioSourceData> new_audio_source_data
	);

  private:
    // global settings for spatial audio
	const SpatialAudioSettings& audio_settings;

	bool audio_device_initialized = false;

	// for playing depth estimation data
	std::vector<ALuint> sources;
	std::vector<std::vector<ALuint>> buffers;
	std::vector<DepthAudioSourceData> depth_audio_sources_data;

	// for playing object detection data
	std::vector<ObjectAudioSourceData> object_audio_sources_data;
	std::vector<short> audio_labels_file_buffer;
	int AUDIO_FILE_SAMPLE_RATE;

	ALCdevice* device;
	ALCcontext* context;

	void setupDepthAudioSources();
	void loadAudioLabelsFile();
};
