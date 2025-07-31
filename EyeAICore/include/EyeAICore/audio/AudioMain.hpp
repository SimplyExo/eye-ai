#pragma once

#include "EyeAICore/audio/AudioData.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include "EyeAICore/audio/Source.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <iostream>
#include <memory>

class AudioMain {
  public:
	void setupAudioDevice();
	void destroyAudioDevice();
	//void playSound(float frequency, float duration);

  private:
	std::unique_ptr<Source> source;
	ALCdevice* device;
	ALCcontext* context;
	
};