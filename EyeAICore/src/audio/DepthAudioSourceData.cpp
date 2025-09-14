#include "EyeAICore/audio/DepthAudioSourceData.hpp"
#include <cmath>
#include <numbers>
#include <vector>

std::vector<short>
createAudioData(float frequency, float duration, int sample_rate) {
	// constant parameters
	const float PI = std::numbers::pi;
	const int numSamples =
		static_cast<int>(static_cast<double>(sample_rate) * duration);
	//const float fade_out_duration = 0.01;
	//const int fade_out_samples =
		//static_cast<int>(static_cast<double>(sample_rate) * fade_out_duration);
	const float amplitude = 0.97f;

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
		const float t = static_cast<float>(i) / static_cast<float>(sample_rate);
		samples[i] = static_cast<short>(
			amplitude * 32760 * std::sin(2.0f * PI * frequency * t)
		);
	}
	/*
	 float m = -(static_cast<float>(samples[numSamples - fade_out_samples - 1]) / (fade_out_duration));
float c = -(m * duration);

for (int i = numSamples - fade_out_samples; i < numSamples; i++) {
   float t = static_cast<float>(i) / static_cast<float>(sample_rate);
   float value = (m * t) + c;
   samples[i] = static_cast<short>(value);
}
	 */


	return samples;
}