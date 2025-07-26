#include "EyeAICore/audio/AudioData.hpp"
#include "EyeAICore/audio/Source.hpp"
#include <iostream>
#include <AL/al.h>
#include <AL/alc.h>
#include <array>
#include <thread>

Source::Source(const AudioData& data,  const std::array<float, 3>& position) : audioData(data){
    this-> x_pos = position[0];
    this-> y_pos = position[1];
    this-> z_pos = position[2];
    generateBuffer();
    generateSource();
    playSource();

}

void Source::generateBuffer(){
    int buffer_size = audioData.numSamples * sizeof(short);
    alGenBuffers(1, &buffer);
    alBufferData(buffer, AL_FORMAT_MONO16, audioData.samples.data(),buffer_size, audioData.sampleRate);
}

void Source::generateSource(){
    alGenSources(1, &source);
    alSourcei(source, AL_BUFFER, buffer);
    alSource3f(source, AL_POSITION, x_pos, y_pos, z_pos);
}

void Source::playSource(){
    alSourcePlay(source);
    ALint state;
    do {
        alGetSourcei(source, AL_SOURCE_STATE, &state);
    } while (state == AL_PLAYING);
}

Source::~Source(){
    alDeleteSources(1, &source);
    alDeleteBuffers(1, &buffer);
}





