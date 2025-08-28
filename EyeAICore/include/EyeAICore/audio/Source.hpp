#pragma once

/*
Neben-Instanz, kontrolliert Sources
*/
#include "AudioData.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <iostream>

/*
This class generates a OpenAl-soft source, which will play a sound.
The sound played will be determined by the passed audiodata, and
the origin of the sound by the passed position in 3d space.
*/

class Source {
  public:
	Source(const AudioData& audioData, const std::array<float, 3>& position);
	~Source();

  private:
	//position of the Source
	float x_pos;
	float y_pos;
	float z_pos;

	ALuint buffer;
	ALuint source;
	const AudioData& audioData;

	void generateBuffer();
	void generateSource();
	void fillBuffer();
	void playSource();
};