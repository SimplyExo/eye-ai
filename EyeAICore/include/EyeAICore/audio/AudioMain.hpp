#pragma once
#include "EyeAICore/audio/AudioData.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include "EyeAICore/audio/Source.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <iostream>

class AudioMain {
  public:
	AudioMain();
	~AudioMain();
	void playSound(float frequency, float duration);

  private:
	ALCdevice* device;
	ALCcontext* context;
	
};