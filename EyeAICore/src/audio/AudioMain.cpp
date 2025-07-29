#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/AudioData.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include "EyeAICore/audio/Source.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <iostream>

AudioMain::AudioMain() {
	// Opening the default audio device
	device = alcOpenDevice(NULL);
	if (!device) {
		std::cout << "Das Audiogerät konnte nicht geöffnet werden."
				  << std::endl;
	}

	// creating and attatching the context to the device, make the context the
	// current one
	context = alcCreateContext(device, NULL);
	if (!alcMakeContextCurrent(context)) {
		std::cout << "Fehler bei Context" << std::endl;
	}

	// setting position of Listener to (0|0|0)
	alListener3f(AL_POSITION, 0.0, 0.0, 0.0);
}

void AudioMain::playSound(float frequency, float duration) {
	AudioData audioData1(frequency, duration);
	std::array<float, 3> position = {-1.0, 0.0, 0.0};
	Source(audioData1, position);
}

AudioMain::~AudioMain() {
	// cleaning up
	alcMakeContextCurrent(nullptr);
	alcDestroyContext(context);
	alcCloseDevice(device);
}
