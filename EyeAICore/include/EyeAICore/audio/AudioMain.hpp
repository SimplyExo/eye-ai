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

#include "EyeAICore/audio/AudioSourceData.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <atomic>
#include <vector>

class AudioMain {
  public:
	AudioMain();
	~AudioMain();
	void startAudioLoop(std::atomic<bool>& running);
	void changeAudioData(std::vector<AudioSourceData> new_audio_source_data);

  private:
	bool audio_device_initialized = false;
	const float BUFFER_DURATION = 1; //in seconds
	const int NUMBER_OF_SOURCES = 16;
	const int BUFFERS_PER_SOURCE = 3;

	std::vector<ALuint> sources;
	std::vector<std::vector<ALuint>> buffers;
	std::vector<AudioSourceData> audio_sources_data;

	ALCdevice* device;
	ALCcontext* context;

	void setupSources();
};

