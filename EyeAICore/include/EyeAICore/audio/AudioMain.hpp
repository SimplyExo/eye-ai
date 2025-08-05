#pragma once

#include "EyeAICore/audio/AudioData.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include "EyeAICore/audio/Source.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <iostream>
#include <memory>
#include <array>

class AudioMain {
  public:
	AudioMain();
	~AudioMain();
	void playSound(float frequency, float duration, std::array<float,3> position);

  private:
	std::unique_ptr<Source> source;
	ALCdevice* device;
	ALCcontext* context;
	
};

