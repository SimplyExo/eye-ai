#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/ByteArrayParser.hpp"
#include "EyeAICore/audio/DepthAudioSourceData.hpp"
#include "EyeAICore/audio/ObjectAudioSourceData.hpp"
#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include "sndfile.h"
#include <AL/al.h>
#include <AL/alc.h>
#include <atomic>
#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <sndfile.hh>
#include <string>
#include <thread>
#include <vector>

#define ALC_HRTF_SOFT 0x1992

typedef ALCboolean(ALC_APIENTRY* LPALCRESETDEVICESOFT)(
	ALCdevice*,
	const ALCint*
);

AudioMain::AudioMain(const SpatialAudioSettings& audio_settings)
	: audio_settings(audio_settings) {
	/*
	Initialises audio playback:
	- Prepares vectors in which the sources, buffers and AudioSourceData
	  is stored
	- Creates OpenAl device and context, for audio playback
	- Enabling HRTF (head-related transfer function) if possible
	*/

	// Preparing the vectors
	buffers.resize(
		audio_settings.NUMBER_OF_SOURCES,
		std::vector<ALuint>(audio_settings.BUFFERS_PER_SOURCE)
	);
	sources.resize(audio_settings.NUMBER_OF_SOURCES);
	depth_audio_sources_data.resize(
		audio_settings.NUMBER_OF_SOURCES,
		DepthAudioSourceData{
			200.0f, 1.0f, audio_settings.SAMPLE_RATE, 0.0f, 0.0f, 0.0f
		}
	);

	// Setting up the OpenAL device configuration
	device = alcOpenDevice(NULL);
	if (!device) {
		std::cout << "Das Audiogerät konnte nicht geöffnet werden.\n";
		return;
	}
	/*
	// Checking for HRTF support
	if (alcIsExtensionPresent(device, "ALC_SOFT_HRTF") == ALC_TRUE) {
		audio_settings.logInfoCallback(
			"[AudioMain] HRTF present, activating ..."
		);

		// Retrieving necessary function
		LPALCRESETDEVICESOFT alcResetDeviceSOFT = (LPALCRESETDEVICESOFT)
			alcGetProcAddress(device, "alcResetDeviceSOFT");

		// Setting up Attributes for HRTF
		ALCint attribs[] = {ALC_HRTF_SOFT, ALC_TRUE, 0};
		alcResetDeviceSOFT(device, attribs);

	} else {
		audio_settings.logInfoCallback("[AudioMain] HRTF not present");
	}
	*/

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

void AudioMain::startDepthAudioLoop(std::atomic<bool>& running) {
	/*
	Constantly cycles through all sources:
	- Unqueue played buffers
	- Fill them up with new data
	- Queues buffers back
	- Update source position
	- Restart stopped sources
	Therefore ensuring continues playback
	*/

	audio_settings.logInfoCallback(
		"[DepthAudioLoop] Starting depth audio loop..."
	);

	if (!audio_device_initialized) {
		audio_settings.logInfoCallback(
			"[DepthAudioLoop] Audio device not initialized. Aborting ..."
		);
		return;
	}

	setupSources();

	while (running) {
		if (audio_settings.depth_audio_paused) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			continue;
		}
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
					buf, AL_FORMAT_MONO16,
					depth_audio_sources_data[i].samples.data(),
					depth_audio_sources_data[i].number_of_samples *
						sizeof(short),
					depth_audio_sources_data[i].sample_rate
				);
				alSourceQueueBuffers(source, 1, &buf);
				// Updating the source's position
				alSource3f(
					source, AL_POSITION,
					depth_audio_sources_data[i].x1_position,
					depth_audio_sources_data[i].x2_position,
					depth_audio_sources_data[i].x3_position
				);
				processed--;
			}

			alSourcef(source, AL_MAX_DISTANCE, 1.5f);
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
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

void AudioMain::startObjectAudioLoop(std::atomic<bool>& running) {
	loadAudioLabelsFile();

	ALuint source;
	ALuint buffer;
	std::vector<short> sound_buffer;
	int MODIFIED_SAMPLE_RATE = AUDIO_FILE_SAMPLE_RATE / 1000; // adapt to ms

	alGenSources(1, &source);
	alGenBuffers(1, &buffer);
	std::vector<ObjectAudioSourceData> object_audio_sources_data_copy;
	while (running) {
		if (audio_settings.object_audio_paused) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			continue;
		}
		object_audio_sources_data_copy = object_audio_sources_data;
		for (auto audio_data : object_audio_sources_data_copy) {
			sound_buffer.resize(
				MODIFIED_SAMPLE_RATE *
				(audio_data.sound_end - audio_data.sound_begin)
			);
			std::copy(
				audio_labels_file_buffer.begin() +
					(MODIFIED_SAMPLE_RATE * audio_data.sound_begin),
				audio_labels_file_buffer.begin() +
					(MODIFIED_SAMPLE_RATE * audio_data.sound_end),
				sound_buffer.begin()
			);

			alBufferData(
				buffer, AL_FORMAT_MONO16, sound_buffer.data(),
				sound_buffer.size() * sizeof(short), AUDIO_FILE_SAMPLE_RATE
			);
			alSourcei(source, AL_BUFFER, buffer);

			alSourcePlay(source);

			ALint source_state = AL_PLAYING;
			while (source_state == AL_PLAYING) {
				alGetSourcei(source, AL_SOURCE_STATE, &source_state);
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
			alSourceStop(source);
			alSourcei(source, AL_BUFFER, AL_NONE);
			sound_buffer.clear();
			object_audio_sources_data_copy.clear();
		}
	}

	alDeleteBuffers(1, &buffer);
	alDeleteSources(1, &source);
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
		DepthAudioSourceData source_data = depth_audio_sources_data[i];

		// Generating each buffer, filling it up and queuing it to the source
		alGenBuffers(audio_settings.BUFFERS_PER_SOURCE, buffers[i].data());
		for (auto buf : buffers[i]) {
			alBufferData(
				buf, AL_FORMAT_MONO16,
				depth_audio_sources_data[i].samples.data(),
				depth_audio_sources_data[i].number_of_samples * sizeof(short),
				depth_audio_sources_data[i].sample_rate
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

void AudioMain::loadAudioLabelsFile() {
	audio_settings.logInfoCallback("[LoadAudioLabelsFile] Started loading ...");

	MemoryData mem{
		audio_settings.coco_labels_audio.data(),
		static_cast<sf_count_t>(audio_settings.coco_labels_audio.size()), 0
	};

	SF_VIRTUAL_IO vio;
	vio.get_filelen = vio_get_filelen;
	vio.seek = vio_seek;
	vio.read = vio_read;
	vio.write = vio_write;
	vio.tell = vio_tell;

	SF_INFO info{};
	SNDFILE* snd = sf_open_virtual(&vio, SFM_READ, &info, &mem);
	if (!snd) {
		std::cerr << "sf_open_virtual fehlgeschlagen: " << sf_strerror(nullptr)
				  << "\n";
		return;
	}

	// --- Audioinfos auslesen ---
	AUDIO_FILE_SAMPLE_RATE = info.samplerate;
	audio_settings.logInfoCallback(
		std::format("File sample rate: {}", info.samplerate)
	);
	audio_settings.logInfoCallback(std::format("Format: {}", info.format));
	audio_settings.logInfoCallback(std::format("Channels: {}", info.channels));

	// --- Daten einlesen ---
	audio_labels_file_buffer.resize(info.frames * info.channels);
	sf_count_t read_frames =
		sf_readf_short(snd, audio_labels_file_buffer.data(), info.frames);

	if (read_frames <= 0) {
		audio_settings.logErrorCallback(
			"[LoadAudioLabelsFile] Could not load file into memory"
		);
	}

	// --- Aufräumen ---
	sf_close(snd);

	audio_settings.logInfoCallback(
		"[LoadAudioLabelsFile] Finished loading ..."
	);
}

void AudioMain::changeDepthAudioData(
	std::vector<DepthAudioSourceData> new_audio_source_data
) {
	this->depth_audio_sources_data = new_audio_source_data;
}

void AudioMain::changeObjectAudioData(
	std::vector<ObjectAudioSourceData> new_audio_source_data
) {
	this->object_audio_sources_data = new_audio_source_data;
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