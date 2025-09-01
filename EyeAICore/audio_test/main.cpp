#include "EyeAICore/audio/SpatialAudio.hpp"
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

	SpatialAudio spatialAudio;

	std::this_thread::sleep_for(std::chrono::seconds(10));
	
	/*
	ALCdevice* device;
	ALCcontext* context;

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

	ALuint source;
	ALuint buffer;

	alGenSources(1, &source);
	alGenBuffers(1, &buffer);

	AudioSourceData audio_data{200.0f, 5.0f, 0.0f,0.0f,0.0f};

	alBufferData(buffer,AL_FORMAT_MONO16 ,audio_data.samples.data(), audio_data.number_of_samples * sizeof(short), audio_data.sample_rate);
	
	alDistanceModel(AL_LINEAR_DISTANCE);

	alSourcei(source, AL_BUFFER, buffer);
	alSource3f(source, AL_POSITION, 1.5f,1.5f,0.0f);
	alSourcei(source, AL_REFERENCE_DISTANCE, 0.0f);
	alSourcei(source, AL_MAX_DISTANCE, 3.0f);
	alSourcePlay(source);
	

	std::this_thread::sleep_for(std::chrono::seconds(5));
	*/
	return 0;
}



