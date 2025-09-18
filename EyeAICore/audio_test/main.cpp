#include <AL/al.h>
#include <AL/alc.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <numbers>
#include <math.h>

std::vector<short> createData(float duration, int sample_rate, float base_frequency);

int main() {
    std::cout << "Inside Main.cpp1" << std::endl;

    ALCcontext* context;
    ALCdevice* device;

    device = alcOpenDevice(NULL);
    if (!device) {
        std::cout << "Device" << std::endl;
        return -1;
    }

    context = alcCreateContext(device, NULL);
    alcMakeContextCurrent(context);

    // Listener Position setzen (wichtig!)
    alListener3f(AL_POSITION, 0.0f, 0.0f, 0.0f);
    ALfloat orientation[] = {0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
    alListenerfv(AL_ORIENTATION, orientation);

    ALuint source;
    ALuint buffer;

    alGenBuffers(1, &buffer);
    alGenSources(1, &source);

    // Korrigierte Parameter
    const int SAMPLE_RATE = 48000;
    std::vector<short> samples = createData(0.02f, SAMPLE_RATE, 500);
    std::cout << samples.size() << std::endl;
    
    alBufferData(buffer, AL_FORMAT_MONO16, samples.data(), 
                samples.size() * sizeof(short), SAMPLE_RATE);

    alSourcei(source, AL_BUFFER, buffer);
    alSourcef(source, AL_GAIN, 1.0f);  // Volle Lautstärke
    alSource3f(source, AL_POSITION, 0.0f, 0.0f, 0.0f);

    alSourcePlay(source);

    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Cleanup
    alDeleteSources(1, &source);
    alDeleteBuffers(1, &buffer);
    alcMakeContextCurrent(nullptr);
    alcDestroyContext(context);
    alcCloseDevice(device);

    return 0;
}

std::vector<short> createData(float duration, int sample_rate, float base_frequency) {
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