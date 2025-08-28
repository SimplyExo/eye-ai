#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/AudioSourceData.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <iostream>
#include <thread>
#include <vector>

AudioMain::AudioMain() {
	/*
	Initialises audio playback:
	- Prepares vectors in which the sources, buffers and AudioSourceData
	  is stored
	- Creates OpenAl device and context, for audio playback
	*/

	// Preparing the vectors
	buffers.resize(NUMBER_OF_SOURCES, std::vector<ALuint>(BUFFERS_PER_SOURCE));
	sources.resize(NUMBER_OF_SOURCES);
	audio_sources_data.resize(
		NUMBER_OF_SOURCES, AudioSourceData{25.0f, 1.0f, 0.0f,0.0f,0.0f}
	);

	// Setting up the OpenAL configuration
	device = alcOpenDevice(nullptr);
	if (!device) {
		std::cout << "Das Audiogerät konnte nicht geöffnet werden.\n";
	}
	context = alcCreateContext(device, nullptr);
	if (!alcMakeContextCurrent(context)) {
		std::cout << "Fehler bei Context.\n";
	}

	// Setting the listener to his default position of (0|0|0)
	alListener3f(AL_POSITION, 0.0, 0.0, 0.0);
}

void AudioMain::startAudioLoop(std::atomic<bool>& running) {
	/*
	Constantly cycles through all sources:
	- Unqueue played buffers
	- Fill them up with new data
	- Queues buffers back
	- Update source position
	- Restart stopped sources
	Therefore ensuring continues playback
	*/

	setupSources();

	while (running) {
		for (int i = 0; i < NUMBER_OF_SOURCES; i++) {
			// Retrieving the source and it's AudioSourceData
			auto& source = sources[i];

			/*
			Checking if a buffer has been fully played.
			If so, it will be filled up again.
			*/
			ALint processed = 0;
			alGetSourcei(source, AL_BUFFERS_PROCESSED, &processed);
			if (processed > 0) {
				// Unqueueing the buffer, filling it up, and requeueing it
				ALuint buf;
				alSourceUnqueueBuffers(source, 1, &buf);
				alBufferData(
					buf, AL_FORMAT_MONO16, audio_sources_data[i].samples.data(),
					audio_sources_data[i].number_of_samples * sizeof(short), audio_sources_data[i].sample_rate
				);
				alSourceQueueBuffers(source, 1, &buf);
				// Updating the source's position
				alSource3f(
					source, AL_POSITION, audio_sources_data[i].x1_position,
					audio_sources_data[i].x2_position,
					audio_sources_data[i].x3_position
				);
				processed--;
			}

			// Restart the source if it has stopped
			ALint state = AL_PAUSED;
			alGetSourcei(source, AL_SOURCE_STATE, &state);
			if (state == AL_STOPPED) {
				alSourcePlay(source);
			}
			
		}

		// To reduce program-load, the loop pauses
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

void AudioMain::setupSources() {
	/*
	Preparing the sources and buffers for playback:
	- Generating the sources
	- Generating the buffers and filling them up with AudioData,
	  according to the AudioSourceData specifications
	- Queuing the buffers to the source
	- Setting the right position for the source, according to the
	  AudioSourceData specifications
	*/

	alGenSources(NUMBER_OF_SOURCES, sources.data());

	/*
	This loop handles the buffers and position of each source
	*/
	for (int i = 0; i < NUMBER_OF_SOURCES; ++i) {
		// Extracting the AudioSourceData for the source, and creating according
		// AudioData
		AudioSourceData source_data = audio_sources_data[i];
	

		// Generating each buffer, filling it up and queuing it to the source
		alGenBuffers(BUFFERS_PER_SOURCE, buffers[i].data());
		for (auto buf : buffers[i]) {
			alBufferData(
					buf, AL_FORMAT_MONO16, audio_sources_data[i].samples.data(),
					audio_sources_data[i].number_of_samples * sizeof(short), audio_sources_data[i].sample_rate
				);
			alSourceQueueBuffers(sources[i], 1, &buf);
		}
		// Setting the right position for the source
		alSource3f(
			sources[i], AL_POSITION, source_data.x1_position,
			source_data.x2_position, source_data.x3_position
		);
		alSourcePlay(sources[i]);
	}
}

void AudioMain::changeAudioData(std::vector<AudioSourceData> new_audio_source_data) {
	this->audio_sources_data = new_audio_source_data;
}

AudioMain::~AudioMain() {
	/*
	Ensures proper resource management:
	- deletes sources and buffers
	- properly ends context and device
	*/
	alDeleteSources(NUMBER_OF_SOURCES, sources.data());
	for (auto buff : buffers) {
		alDeleteBuffers(BUFFERS_PER_SOURCE, buff.data());
	}
	alcMakeContextCurrent(nullptr);
	alcDestroyContext(context);
	alcCloseDevice(device);
}
