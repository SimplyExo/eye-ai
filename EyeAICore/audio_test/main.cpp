#include "EyeAICore/audio/SpacialAudio.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <thread>
#include <vector>

void audio_loop(std::vector<ALuint>& sources);

int frame_number = 0;
int audio_frame = 0;
int main() {
	std::cout << "Inside Main.cpp" << std::endl;

	SpacialAudio spacialAudio;

	std::this_thread::sleep_for(std::chrono::seconds(10));
	
	/*
	ALCdevice* device;
	ALCcontext* context;

	const int NUMBER_OF_SOURCES = 16;
	const int BUFFERS_PER_SOURCE = 3;
	// Opening the default audio device
	device = alcOpenDevice(nullptr);
	if (!device) {
		std::cout << "Das Audiogerät konnte nicht geöffnet werden.\n";
	}

	// creating and attatching the context to the device, make the context the
	// current one
	context = alcCreateContext(device, nullptr);
	if (!alcMakeContextCurrent(context)) {
		std::cout << "Fehler bei Context.\n";
	}

	// setting position of Listener to (0|0|0)
	alListener3f(AL_POSITION, 0.0, 0.0, 0.0);

	std::vector<ALuint> sources(NUMBER_OF_SOURCES);
	alGenSources(NUMBER_OF_SOURCES, sources.data());

	AudioData audioData(200.0f, 1.00f);

	std::vector<std::vector<ALuint>> buffers(
		NUMBER_OF_SOURCES, std::vector<ALuint>(BUFFERS_PER_SOURCE)
	);
	for (size_t i = 0; i < NUMBER_OF_SOURCES; ++i) {
		alGenBuffers(BUFFERS_PER_SOURCE, buffers[i].data());
		for (auto buf : buffers[i]) {
			alBufferData(
				buf, AL_FORMAT_MONO16, audioData.samples.data(),
				audioData.numSamples * sizeof(short), audioData.sampleRate
			);
			alSourceQueueBuffers(sources[i], 1, &buf);
		}
		alSourcePlay(sources[i]);
	}

	audio_loop(sources);
	*/
	
	
	return 0;
}
/*
void audio_loop(std::vector<ALuint>& sources) {
	std::cout << "Inside audio_loop" << "\n";
	AudioData audioData(200.0f, 1.0f);
	while (true) {
		for (auto& source : sources) {
			std::cout << "Audio Frame: " << audio_frame << "\n";

			ALint processed = 0;
			ALint queued = 0;
			ALint state = AL_PAUSED;

			alGetSourcei(source, AL_BUFFERS_PROCESSED, &processed);
			std::cout << "Buffers processed: " << processed << "\n";

			alGetSourcei(source, AL_BUFFERS_QUEUED, &queued);
			std::cout << "Buffers queued: " << queued << "\n";

			if (processed > 0) {
				
				ALuint buf;
				alSourceUnqueueBuffers(source, 1, &buf);
				alBufferData(
					buf, AL_FORMAT_MONO16, audioData.samples.data(),
					audioData.numSamples * sizeof(short), audioData.sampleRate
				);
				alSourceQueueBuffers(source, 1, &buf);
				processed--;
			}

			alGetSourcei(source, AL_SOURCE_STATE, &state);
			std::cout << "Source state: " << state << "\n";
			if (state == AL_STOPPED) {
				alSourcePlay(source);
			}
			std::cout << std::endl;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		std::cout << std::endl;
		audio_frame++;
	}
}
*/
