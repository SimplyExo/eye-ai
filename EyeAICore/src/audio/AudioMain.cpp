#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/AudioData.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include "EyeAICore/audio/Source.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <iostream>

void AudioMain::setupAudioDevice() {
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
	source = std::make_unique<Source>(AudioData(200.0f, 10.0f), std::array<float,3>{-1.0,0.0,0.0});
}
/*
void AudioMain::playSound(float frequency, float duration) {
	AudioData audioData1(frequency, duration);
	std::array<float, 3> position = {-1.0, 0.0, 0.0};
	Source(audioData1, position);
}

*/

void AudioMain::destroyAudioDevice() {
	// cleaning up
	source.reset();
	alcMakeContextCurrent(nullptr);
	alcDestroyContext(context);
	alcCloseDevice(device);
}
