#pragma once

#include "EyeAICore/audio/DepthAudioSourceData.hpp"
#include "EyeAICore/audio/ObjectAudioSourceData.hpp"
#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_set>
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
	explicit AudioMain(const SpatialAudioSettings& audio_settings);
	~AudioMain();

	// Functions for playing depth estimation data
	void startDepthAudioLoop(std::atomic<bool>& running);
	void changeDepthAudioData(
		std::vector<DepthAudioSourceData> new_audio_source_data
	);

	// Functions for playing object detection data
	void startObjectAudioLoop(std::atomic<bool>& running);
	void changeObjectAudioData(
		const std::vector<ObjectAudioSourceData>& new_audio_source_data
	);

  private:
    // global settings for spatial audio
	const SpatialAudioSettings& audio_settings;

	// for playing depth estimation data
	std::vector<ALuint> sources;
	std::vector<std::vector<ALuint>> buffers;
	std::vector<DepthAudioSourceData> depth_audio_sources_data;

	// for playing object detection data
	std::queue<ObjectAudioSourceData> object_audio_sources_data;
	std::unordered_set<int> seen_objects;
	std::vector<short> audio_labels_file_buffer;
	std::mutex object_mutex;
	int AUDIO_FILE_SAMPLE_RATE = 0;

	ALCdevice* device = AL_NONE;
	ALCcontext* context = AL_NONE;

	void setupDepthAudioSources();
	void loadAudioLabelsFile();
};
