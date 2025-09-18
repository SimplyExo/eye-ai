#include "EyeAICore/audio/DepthAudioSourceData.hpp"
#include <cmath>
#include <numbers>
#include <vector>

std::vector<short>
createAudioData(float base_frequency, float duration, int sample_rate) {
	const float PI = std::numbers::pi;
    const int numSamples = static_cast<int>(static_cast<double>(sample_rate) * duration);
    const float amplitude = 0.8f;
    const float decay_rate = 8.0f; // Steuert wie schnell der Klick abklingt
    
    std::vector<short> samples(numSamples);
    
    for (int i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        
        // Exponentieller Decay für natürlichen Klick-Sound
        float envelope = std::exp(-decay_rate * t);
        
        // Mischung aus Grundfrequenz und Harmonischen für prägnanten Sound
        float wave = std::sin(2.0f * PI * base_frequency * t) +
                     0.5f * std::sin(2.0f * PI * base_frequency * 2.0f * t) +
                     0.25f * std::sin(2.0f * PI * base_frequency * 3.0f * t);
        
        samples[i] = static_cast<short>(
            amplitude * envelope * wave * 32760 / 1.75f // Normalisierung wegen der Harmonischen
        );
    }
    
    return samples;
}