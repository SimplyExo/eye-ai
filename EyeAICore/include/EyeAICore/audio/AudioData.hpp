#pragma once

#include <cmath>
#include <iostream>
#include <vector>

class AudioData {
  public:
	const int sampleRate = 44100;
	int numSamples;
	float frequency;
	float duration;
	std::vector<short> samples;

	AudioData(float frequency, float duration);

  private:
	const float PI = 3.14159265f;
	const float amplitude = 1.0f;
	std::vector<short> getAudioTone();
};