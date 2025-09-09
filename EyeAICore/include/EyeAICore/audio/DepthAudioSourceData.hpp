#pragma once

#include <vector>

/*
Struct of all the data a source needs to play a specific
sound at a specific location
*/

// Function creating the the samples the source plays
std::vector<short>
createAudioData(float frequency, float duration, int sample_rate);

struct DepthAudioSourceData {
	float frequency;
	float duration;
	int sample_rate;
	int number_of_samples;
	float x1_position;
	float x2_position;
	float x3_position;
	std::vector<short> samples;

	DepthAudioSourceData(
		float freq,
		float dur,
		int sample_rate,
		float x1,
		float x2,
		float x3
	)
		: frequency(freq), duration(dur), sample_rate(sample_rate),
		  number_of_samples(
			  static_cast<int>(static_cast<double>(sample_rate) * dur)
		  ),
		  x1_position(x1), x2_position(x2), x3_position(x3),
		  samples(createAudioData(freq, dur, sample_rate)) {}
};
