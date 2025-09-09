#include "EyeAICore/audio/SpatialAudio.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>
#include <vector>
#include <sndfile.hh>
#include <nlohmann/json.hpp>
#include <unordered_map>

#define ALC_HRTF_SOFT 0x1992

int frame_number = 0;
int audio_frame = 0;
int main() {
	std::cout << "Inside Main.cpp1" << std::endl;

	ALCcontext* context;
	ALCdevice* device;

	device = alcOpenDevice(NULL);

	if (alcIsExtensionPresent(device, "ALC_SOFT_HRTF") == ALC_TRUE) {
		typedef ALCboolean(ALC_APIENTRY * LPALCRESETDEVICESOFT)(
			ALCdevice*, const ALCint*
		);
		LPALCRESETDEVICESOFT alcResetDeviceSOFT = (LPALCRESETDEVICESOFT)
			alcGetProcAddress(device, "alcResetDeviceSOFT");
		std::cout << "HRTF\n";
		ALCint attribs[] = {ALC_HRTF_SOFT, ALC_TRUE, 0};
		alcResetDeviceSOFT(device, attribs);
	}

	context = alcCreateContext(device, NULL);
	alcMakeContextCurrent(context);

	ALCint hrtf_state;
	alcGetIntegerv(device, ALC_HRTF_SOFT, 1, &hrtf_state);

	if (hrtf_state == ALC_TRUE) {
		std::cout << "HRTF is active\n";
	} else {
		std::cout << "HRTF is not active\n";
	}

	std::cout << "-------------------------------------" << std::endl;

	ALuint source;
	ALuint buffer;

	

	SndfileHandle file;
	
	file = SndfileHandle("../../../EyeAIApp/app/src/main/assets/coco_labels.wav");

	if(file.error()){
		std::cout << "Error: "<< file.strError() << std::endl;
	}

	std::cout << "Sample rate: " << file.samplerate() << "\n";
	std::cout << "Format " << file.format() << "\n"; 
	std::cout << "Channels: " << file.channels() << "\n";

	std::vector<short> file_buffer(file.frames());

	int SAMPLE_RATE  = file.samplerate() / 1000;
	std::vector<short> sound_buffer(SAMPLE_RATE * (95140 - 94610)); 
	

	file.read(file_buffer.data(), file_buffer.size());


	std::ifstream f("../../../EyeAIApp/app/src/main/assets/coco_labels_data.json");

	nlohmann::json datas = nlohmann::json::parse(f);
	std::unordered_map<std::string,std::array<int, 2>> label_data;
	for(const auto& data: datas){
		label_data[data["text"]] = {data["start"], data["end"]};
	}

	std::cout << label_data.at("chair")[0] << std::endl;
	std::cout << label_data.at("chair")[1] << std::endl;
	std::copy(file_buffer.begin() + (SAMPLE_RATE * 94610), file_buffer.begin() + (SAMPLE_RATE * 95140), sound_buffer.begin());

	

	alGenBuffers(1, &buffer);
	alGenSources(1, &source);

	std::cout << file_buffer.data() << std::endl;

	alBufferData(buffer, AL_FORMAT_MONO16, sound_buffer.data(), sound_buffer.size() * sizeof(short) , file.samplerate());

	alSourcei(source, AL_BUFFER, buffer);

	alSourcePlay(source);
	// SpatialAudio spatialAudio;
	std::this_thread::sleep_for(std::chrono::seconds(35));

	return 0;
}
