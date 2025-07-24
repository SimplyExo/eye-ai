#include <iostream>
#include <AL/al.h>
#include <AL/alc.h>
#include <cstring>
#include <vector>
#include <cmath>

constexpr float PI = 3.14159265f;

int audio_main(){
    std::cout << "Inside audio_main.cpp" << std::endl;
    ALCdevice* device;
    ALCcontext* context;

    //Opening the default audio device
    device = alcOpenDevice(NULL);
    if(!device){
        std::cout << "Das Audiogerät konnte nicht geöffnet werden." << std::endl;
    }

    context = alcCreateContext(device, NULL);
    if (!alcMakeContextCurrent(context)){
        std::cout << "Fehler bei Context" << std::endl;
    }

    alListener3f(AL_POSITION, 0.0,0.0,0.0);

    // Audio-Parameter
    const int sampleRate = 44100;
    const float frequency = 200.0f; // 200 Hz
    const float duration = 2.0f;    // Sekunden
    const int numSamples = static_cast<int>(sampleRate * duration);
    const float amplitude = 0.5f;   // Lautstärke (0.0–1.0)

    // PCM-Daten erzeugen (Mono, 16-bit)
    std::vector<short> samples(numSamples);
    for (int i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / sampleRate;
        samples[i] = static_cast<short>(amplitude * 32760 * std::sin(2.0f * PI * frequency * t));
    }

    // Buffer erzeugen
    ALuint buffer;
    alGenBuffers(1, &buffer);
    alBufferData(buffer, AL_FORMAT_MONO16, samples.data(), numSamples * sizeof(short), sampleRate);

    // Source erzeugen
    ALuint source;
    alGenSources(1, &source);
    alSourcei(source, AL_BUFFER, buffer);

    alSource3f(source, AL_POSITION, 5.0,5.0,0.0);

    alSourcePlay(source);

    // Warten, bis Ton abgespielt wurde
    ALint state;
    do {
        alGetSourcei(source, AL_SOURCE_STATE, &state);
    } while (state == AL_PLAYING);

    // Aufräumen
    alDeleteSources(1, &source);
    alDeleteBuffers(1, &buffer);
    alcMakeContextCurrent(nullptr);
    alcDestroyContext(context);
    alcCloseDevice(device);
    return 2;
}