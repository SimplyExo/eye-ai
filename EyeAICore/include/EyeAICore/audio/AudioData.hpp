#pragma once

#include <cmath>
#include <iostream>
#include <vector>

/*
This class generates the data of a sound with a specific
frequency and a specific duration. A instance of this class
can be used to create an instance of the Source class, which
will then play the sound
*/

class AudioData {
  public:
	const int sampleRate = 44100;
	int numSamples;
	float frequency;
	float duration;
	std::vector<short> samples;

	AudioData(float frequency, float duration);

  private:
	std::vector<short> getAudioTone();
};