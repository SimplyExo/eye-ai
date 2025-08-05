#include <iostream>
#include "EyeAICore/audio/AudioMain.hpp" 
#include "EyeAICore/audio/SpacialAudio.hpp"
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

int main(){
    std::cout << "Inside Main.cpp" << std::endl;
    
    SpacialAudio spacial_audio;
    spacial_audio.processDepthEstimationData();

    
    return 0;
}