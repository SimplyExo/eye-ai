#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/AudioSettings.hpp"
#include "EyeAICore/audio/AudioSourceData.hpp"
#include "EyeAICore/audio/SpatialAudio.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define ALC_HRTF_SOFT 0x1992

typedef ALCboolean(ALC_APIENTRY* LPALCRESETDEVICESOFT)(
	ALCdevice*,
	const ALCint*
);

AudioMain::AudioMain(const AudioSettings& audio_settings) : audio_settings(audio_settings) {
	/*
	Initialises audio playback:
	- Prepares vectors in which the sources, buffers and AudioSourceData
	  is stored
	- Creates OpenAl device and context, for audio playback
	- Enabling HRTF (head-related transfer function) if possible
	*/

	// Preparing the vectors
	buffers.resize(audio_settings.NUMBER_OF_SOURCES, std::vector<ALuint>(audio_settings.BUFFERS_PER_SOURCE));
	sources.resize(audio_settings.NUMBER_OF_SOURCES);
	audio_sources_data.resize(
		audio_settings.NUMBER_OF_SOURCES, AudioSourceData{200.0f, 1.0f,audio_settings.SAMPLE_RATE, 0.0f, 0.0f, 0.0f}
	);

	// Setting up the OpenAL device configuration
	device = alcOpenDevice(NULL);
	if (!device) {
		std::cout << "Das Audiogerät konnte nicht geöffnet werden.\n";
		return;
	}

	// Checking for HRTF support
	if (alcIsExtensionPresent(device, "ALC_SOFT_HRTF") == ALC_TRUE) {
		audio_settings.logInfoCallback("[AudioMain] HRTF present, activating ...");
		
		// Retrieving necessary function
		LPALCRESETDEVICESOFT alcResetDeviceSOFT = (LPALCRESETDEVICESOFT)
			alcGetProcAddress(device, "alcResetDeviceSOFT");

		// Setting up Attributes for HRTF
		ALCint attribs[] = {ALC_HRTF_SOFT, ALC_TRUE, 0};
		alcResetDeviceSOFT(device, attribs);
		
	} else {
		audio_settings.logInfoCallback("[AudioMain] HRTF not present");
	}

	context = alcCreateContext(device, nullptr);
	if (!alcMakeContextCurrent(context)) {
		std::cout << "Fehler bei Context.\n";
		return;
	}

	audio_device_initialized = true;

	alDistanceModel(AL_LINEAR_DISTANCE);

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

	if (!audio_device_initialized)
		return;

	setupSources();

	while (running) {
		for (int i = 0; i < audio_settings.NUMBER_OF_SOURCES; i++) {
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
					audio_sources_data[i].number_of_samples * sizeof(short),
					audio_sources_data[i].sample_rate
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

			alSourcef(source, AL_MAX_DISTANCE, 1.0f);
			alSourcef(source, AL_ROLLOFF_FACTOR, 1.0f);
			alSourcef(source, AL_REFERENCE_DISTANCE, 0.0f);

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

	alGenSources(audio_settings.NUMBER_OF_SOURCES, sources.data());

	for (auto source : sources) {
		alSourcef(source, AL_MAX_DISTANCE, 1.0f);
		alSourcef(source, AL_ROLLOFF_FACTOR, 1.0f);
		alSourcef(source, AL_REFERENCE_DISTANCE, 0.0f);
	}

	/*
	This loop handles the buffers and position of each source
	*/
	for (int i = 0; i < audio_settings.NUMBER_OF_SOURCES; ++i) {
		// Extracting the AudioSourceData for the source, and creating according
		// AudioData
		AudioSourceData source_data = audio_sources_data[i];

		// Generating each buffer, filling it up and queuing it to the source
		alGenBuffers(audio_settings.BUFFERS_PER_SOURCE, buffers[i].data());
		for (auto buf : buffers[i]) {
			alBufferData(
				buf, AL_FORMAT_MONO16, audio_sources_data[i].samples.data(),
				audio_sources_data[i].number_of_samples * sizeof(short),
				audio_sources_data[i].sample_rate
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

void AudioMain::changeAudioData(
	std::vector<AudioSourceData> new_audio_source_data
) {
	this->audio_sources_data = new_audio_source_data;
}

AudioMain::~AudioMain() {
	/*
	Ensures proper resource management:
	- deletes sources and buffers
	- properly ends context and device
	*/
	alDeleteSources(audio_settings.NUMBER_OF_SOURCES, sources.data());
	for (auto buff : buffers) {
		alDeleteBuffers(audio_settings.BUFFERS_PER_SOURCE, buff.data());
	}
	alcMakeContextCurrent(nullptr);
	alcDestroyContext(context);
	alcCloseDevice(device);
}