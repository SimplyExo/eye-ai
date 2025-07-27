#include "EyeAICore/audio/AudioData.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include "EyeAICore/audio/Source.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

int audio_main() {
	std::cout << "Inside audio_main.cpp" << std::endl;

	std::array<int, 2> pos = {60, 1};
	CalculateSoundOrigin().calculateSoundOrigin(pos, 2.0f);

	ALCdevice* device;
	ALCcontext* context;

	// Opening the default audio device
	device = alcOpenDevice(NULL);
	if (!device) {
		std::cout << "Das Audiogerät konnte nicht geöffnet werden."
				  << std::endl;
		return 1;
	}

	// creating and attatching the context to the device, make the context the
	// current one
	context = alcCreateContext(device, NULL);
	if (!alcMakeContextCurrent(context)) {
		std::cout << "Fehler bei Context" << std::endl;
		return 1;
	}

	// setting position of Listener to (0|0|0)
	alListener3f(AL_POSITION, 0.0, 0.0, 0.0);

	{
		AudioData audioData1(300.0f, 2.0f);
		std::array<float, 3> position = {-1.0, 0.0, 0.0};
		Source source1(audioData1, position);

		AudioData audioData2(200.0f, 3.0f);
		std::array<float, 3> position2 = {1.0, 0.0, 0.0};
		Source source2(audioData2, position2);
	}

	// cleaning up
	alcMakeContextCurrent(nullptr);
	alcDestroyContext(context);
	alcCloseDevice(device);
	return 0;
}