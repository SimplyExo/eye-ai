#include <iostream>
#include "EyeAICore/audio/AudioMain.hpp" 

int main(){
    std::cout << "Inside Main.cpp" << std::endl;
    AudioMain spacialAudio;
    spacialAudio.playSound(200.0f, 2.0f);
    return 0;
}