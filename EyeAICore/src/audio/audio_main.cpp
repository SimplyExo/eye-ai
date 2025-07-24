#include "EyeAICore/audio/AudioData.hpp"
#include "EyeAICore/audio/Source.hpp"
#include <iostream>
#include <AL/al.h>
#include <AL/alc.h>
#include <cstring>
#include <vector>
#include <cmath>
#include <array>

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

    AudioData audioData1(200.0f, 2.0f);
    std::array<float, 3> position = {-1.0,0.0,0.0};
    Source source1(audioData1, position);

    AudioData audioData2(200.0f, 2.0f);
    position = {1.0,0.0,0.0};
    Source source2(audioData2, position);
    

    alcMakeContextCurrent(nullptr);
    alcDestroyContext(context);
    alcCloseDevice(device);
    return 2;
}