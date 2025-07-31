#include <iostream>
#include "EyeAICore/audio/AudioMain.hpp" 
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

int main(){
    std::cout << "Inside Main.cpp" << std::endl;
    AudioMain spacialAudio;
    spacialAudio.setupAudioDevice();
    std::this_thread::sleep_for(2s);
    spacialAudio.destroyAudioDevice();
    return 0;
}