#pragma once

#include "AudioData.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <iostream>

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
	void playSource();
};