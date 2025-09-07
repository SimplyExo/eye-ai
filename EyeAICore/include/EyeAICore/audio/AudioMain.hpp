#pragma once
/*
AudioMain handles the audio playback:
- Creates the audio device
- Creates NUMBER_OF_SOURCES sources
- Creates BUFFERS_PER_SOURCE buffers per source
- Handles the AudioSourceData for each source
Once AudioLoop start, continues playback is ensured,
by refilling empty buffers with new data
*/

#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include "EyeAICore/audio/DepthAudioSourceData.hpp"
#include "EyeAICore/audio/ObjectAudioSourceData.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <atomic>
#include <vector>

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
	bool audio_device_initialized = false;
	const SpatialAudioSettings& audio_settings;

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

	void setupSources();
	void loadAudioLabelsFile();
};
