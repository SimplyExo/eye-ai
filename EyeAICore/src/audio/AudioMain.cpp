#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/ByteArrayParser.hpp"
#include "EyeAICore/audio/DepthAudioSourceData.hpp"
#include "EyeAICore/audio/ObjectAudioSourceData.hpp"
#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "sndfile.h"
#include <AL/al.h>
#include <AL/alc.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <format>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sndfile.hh>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define ALC_HRTF_SOFT 0x1992
#define LOG_INFO(msg) audio_settings.logInfoCallback(msg)
#define LOG_ERROR(msg) audio_settings.logErrorCallback(msg)

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
	- setting OpenAl distance model
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
			0.0f, 1.0f, audio_settings.SAMPLE_RATE, 0.0f, 0.0f, 0.0f
		}
	);

	// Setting up the OpenAL device configuration
	device = alcOpenDevice(NULL);
	if (!device) {
		LOG_INFO("[AudioMain] Could not open audio device.");
		return;
	}
	/*
	// Checking for HRTF support
	if (alcIsExtensionPresent(device, "ALC_SOFT_HRTF") == ALC_TRUE) {
		LOG_INFO(
			"[AudioMain] HRTF present, activating ..."
		);

		// Retrieving necessary function
		LPALCRESETDEVICESOFT alcResetDeviceSOFT = (LPALCRESETDEVICESOFT)
			alcGetProcAddress(device, "alcResetDeviceSOFT");

		// Setting up Attributes for HRTF
		ALCint attribs[] = {ALC_HRTF_SOFT, ALC_TRUE, 0};
		alcResetDeviceSOFT(device, attribs);

	} else {
		LOG_INFO("[AudioMain] HRTF not present");
	}
	*/

	context = alcCreateContext(device, nullptr);
	if (!alcMakeContextCurrent(context)) {
		LOG_ERROR("[AudioMain] Could not open context");
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

	LOG_INFO("[DepthAudioLoop] Starting depth audio loop...");

	if (!audio_device_initialized) {
		LOG_INFO("[DepthAudioLoop] Audio device not initialized. Aborting ...");
		return;
	}

	setupDepthAudioSources();

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
	/*
	Going through all recognized objects:
	- creating a copy of the data
	- loading the right sound for the object
	- calculation position of object
	- waiting until sound is played
	*/

	LOG_INFO("[ObjectAudioLoop] Starting object audio loop");

	// loading the wav file
	loadAudioLabelsFile();

	ALuint source;
	ALuint buffer;
	std::vector<short> sound_buffer;
	// adapting the sample rate to ms for easier use
	int MODIFIED_SAMPLE_RATE = AUDIO_FILE_SAMPLE_RATE / 1000;

	alGenSources(1, &source);
	alGenBuffers(1, &buffer);

	alSourcef(source, AL_GAIN, 0.5f);

	ObjectAudioSourceData object_data;

	while (running) {

		if (audio_settings.object_audio_paused) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			continue;
		}

		{
			PROFILE_AUDIO_FUNCTION()
			LOG_INFO("[ChangeObjectAudioData] Player got lock...");
			std::lock_guard<std::mutex> lock(object_mutex);
			if (object_audio_sources_data.size() == 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
				continue;
			}

			std::queue<ObjectAudioSourceData> temp_queue =
				object_audio_sources_data;
			std::ostringstream oss;

			oss << "[ChangeObjectAudioData] Player Audio Queue (" << temp_queue.size()
				<< " items): ";
			while (!temp_queue.empty()) {
				oss << temp_queue.front().name << " ";
				temp_queue.pop();
			}

			LOG_INFO(oss.str());

			std::string objects_str = "";
			for (const auto& obj_name : seen_objects) {
				objects_str += obj_name + " ";
			}

			LOG_INFO(
				std::format(
					"[ChangeObjectAudioData] Player Seen objects: {}", objects_str
				)
			);

			object_data = object_audio_sources_data.front();
			object_audio_sources_data.pop();
			LOG_INFO("[ChangeObjectAudioData] Player released lock...");
		}

		// preparing the sound to be played, by loading the right portion of
		// the file containing all sounds
		sound_buffer.resize(
			MODIFIED_SAMPLE_RATE *
			(object_data.sound_end - object_data.sound_begin)
		);
		std::copy(
			audio_labels_file_buffer.begin() +
				(MODIFIED_SAMPLE_RATE * object_data.sound_begin),
			audio_labels_file_buffer.begin() +
				(MODIFIED_SAMPLE_RATE * object_data.sound_end),
			sound_buffer.begin()
		);
		LOG_INFO(std::format("[ChangeObjectAudioData] Sound begin of {}: {}", object_data.name, object_data.sound_begin));
		LOG_INFO(std::format("[ChangeObjectAudioData] Sound end of {}: {}", object_data.name, object_data.sound_end));

		// playing the right sound
		alBufferData(
			buffer, AL_FORMAT_MONO16, sound_buffer.data(),
			sound_buffer.size() * sizeof(short), AUDIO_FILE_SAMPLE_RATE
		);
		alSourcei(source, AL_BUFFER, buffer);
		alSource3f(
			source, AL_POSITION, object_data.x1_position,
			object_data.x2_position, object_data.x3_position
		);
		alSourcePlay(source);

		LOG_INFO(
			std::format(
				"[ChangeObjectAudioData] Playing object: {}", object_data.name
			)
		);
		// waiting until the sound is played, so that no sounds overlap
		ALint source_state = AL_PLAYING;
		while (source_state == AL_PLAYING) {
			alGetSourcei(source, AL_SOURCE_STATE, &source_state);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		LOG_INFO(
			std::format(
				"[ChangeObjectAudioData] Played object: {}", object_data.name
			)
		);

		// preparing for next object
		alSourceStop(source);
		alSourcei(source, AL_BUFFER, AL_NONE);
		sound_buffer.clear();
		{
			LOG_INFO("[ChangeObjectAudioData] Play got lock for releasing seen object...");
			std::lock_guard<std::mutex> lock(object_mutex);
			seen_objects.erase(object_data.name);
			LOG_INFO("[ChangeObjectAudioData] Play released lock for releasing seen object...");
		}
	}

	// cleaning up
	alDeleteBuffers(1, &buffer);
	alDeleteSources(1, &source);
}

void AudioMain::setupDepthAudioSources() {
	/*
	Preparing the sources and buffers for playback:
	- Generating the sources
	- Generating the buffers and filling them up with AudioData,
	  according to the AudioSourceData specifications
	- Queuing the buffers to the source
	- Setting the right position for the source, according to the
	  AudioSourceData specifications
	*/

	LOG_INFO("[SetupDepthAudioSources] Setting up depth audio sources ...");

	alGenSources(audio_settings.NUMBER_OF_SOURCES, sources.data());

	for (auto source : sources) {
		alSourcef(source, AL_MAX_DISTANCE, 1.0f);
		alSourcef(source, AL_ROLLOFF_FACTOR, 1.0f);
		alSourcef(source, AL_REFERENCE_DISTANCE, 0.0f);
		alSourcef(source, AL_GAIN, 0.5f);
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
	/*
	Loading the sounds for the objects:
	- making the std::vector<std::byte> readble
	  for libsndfile
	- loading the samples into a vector
	*/
	LOG_INFO("[LoadAudioLabelsFile] Started loading ...");

	MemoryData mem{
		.data = audio_settings.coco_labels_audio.data(),
		.size =
			static_cast<sf_count_t>(audio_settings.coco_labels_audio.size()),
		.pos = 0
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
		LOG_ERROR(
			std::format(
				"[LoadAudioLabelsFile] sf_open_virtual failed: {}",
				sf_strerror(nullptr)
			)
		);
		return;
	}

	// reading audio file information
	AUDIO_FILE_SAMPLE_RATE = info.samplerate;
	LOG_INFO(
		std::format(
			"[LoadAudioLabelsFile] File sample rate: {}", info.samplerate
		)
	);
	LOG_INFO(std::format("[LoadAudioLabelsFile] Format: {}", info.format));
	LOG_INFO(std::format("[LoadAudioLabelsFile] Channels: {}", info.channels));

	// reading the data
	audio_labels_file_buffer.resize(info.frames * info.channels);
	sf_count_t read_frames =
		sf_readf_short(snd, audio_labels_file_buffer.data(), info.frames);

	if (read_frames <= 0) {
		LOG_ERROR("[LoadAudioLabelsFile] Could not load file into memory");
	}

	// cleaning up
	sf_close(snd);

	LOG_INFO("[LoadAudioLabelsFile] Finished loading ...");
}

void AudioMain::changeDepthAudioData(
	std::vector<DepthAudioSourceData> new_audio_source_data
) {
	PROFILE_AUDIO_FUNCTION()	
	this->depth_audio_sources_data = new_audio_source_data;
}

void AudioMain::changeObjectAudioData(
	std::vector<ObjectAudioSourceData> new_audio_source_data
) {
	PROFILE_AUDIO_FUNCTION()
	std::lock_guard<std::mutex> lock(object_mutex);
	LOG_INFO("[ChangeObjectAudioData] Got lock...");

	for (auto new_object : new_audio_source_data) {
		/*
		LOG_INFO(
			std::format(
				"[ChangeObjectAudioData] Queue size: {}",
				object_audio_sources_data.size()
			)
		);

		std::queue<ObjectAudioSourceData> temp_queue =
			object_audio_sources_data;
		std::ostringstream oss;

		oss << "[ChangeObjectAudioData] Audio Queue (" << temp_queue.size()
			<< " items): ";
		while (!temp_queue.empty()) {
			oss << temp_queue.front().name << " ";
			temp_queue.pop();
		}

		LOG_INFO(oss.str());

		std::string objects_str = "Seen objects: ";
		for (const auto& obj_name : seen_objects) {
			objects_str += obj_name + " ";
		}

		LOG_INFO(
			std::format("[ChangeObjectAudioData] Seen objects: {}", objects_str)
		);
		*/
		
		if (!seen_objects.contains(new_object.name) &&
			object_audio_sources_data.size() < 20) {
			LOG_INFO(
				std::format(
					"[ChangeObjectAudioData] New object added: {}",
					new_object.name
				)
			);
			object_audio_sources_data.push(new_object);
			seen_objects.insert(new_object.name);
		} else {
			LOG_INFO(
				std::format(
					"[ChangeObjectAudioData] Did not add object: {}",
					new_object.name
				)
			);
		}
	}
	LOG_INFO("[ChangeObjectAudioData] Released lock...");
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