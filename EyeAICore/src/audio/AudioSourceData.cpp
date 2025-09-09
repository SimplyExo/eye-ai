#include "EyeAICore/audio/DepthAudioSourceData.hpp"
#include <cmath>
#include <vector>
#include <numbers>

std::vector<short>
createAudioData(float frequency, float duration, int sample_rate) {
	// constant parameters
	const float PI = std::numbers::pi;
	const int numSamples =
		static_cast<int>(static_cast<double>(sample_rate) * duration);
	const float amplitude = 1.0f;

	/*
	generating the PCM-Data:
	- t represents the time in the sound, e.g: i = 0 => t = 0 / 44100 = 0s; i
	= 44100 => t = 44100 / 44100 = 1s
	- std::sin(2.0f * PI * frequency * t):
	mathematical sin-function: sin(2πft) adapted
	- amplitude * 32760: sin
	function returns values between -1 and +1, 32760 scales this to
	16-Bit-Audio, amplitude is between 0.0 and 1.0 and adjusts the volume
	*/
	std::vector<short> samples(numSamples);
	for (int i = 0; i < numSamples; ++i) {
		float t = static_cast<float>(i) / static_cast<float>(sample_rate);
		samples[i] = static_cast<short>(
			amplitude * 32760 * std::sin(2.0f * PI * frequency * t)
		);
	}

	return samples;
}