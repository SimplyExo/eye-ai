#include "EyeAICore/audio/SpatialAudio.hpp"
#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/audio/AudioMain.hpp"
#include "EyeAICore/audio/CalculateSoundOrigin.hpp"
#include "EyeAICore/audio/DepthAudioSourceData.hpp"
#include "EyeAICore/audio/ObjectAudioSourceData.hpp"
#include "EyeAICore/audio/SpatialAudioSettings.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <thread>
#include <vector>

#define LOG_INFO(msg) audio_settings.logInfoCallback(msg)
#define LOG_ERROR(msg) audio_settings.logErrorCallback(msg)

SpatialAudio::SpatialAudio(const SpatialAudioSettings& audio_settings)
	: audio_main(audio_settings), audio_settings(audio_settings) {
	readObjectLabelData();
	depth_audio_thread = std::thread([this]() {
		audio_main.startDepthAudioLoop(depth_audio_running);
	});

	object_audio_thread = std::thread([this]() {
		audio_main.startObjectAudioLoop(object_audio_running);
	});
}

void SpatialAudio::getAIData(
	std::span<float, 256 * 256> depth_estimation_data,
	std::vector<ObjectTracker::TrackedBoundingBox> object_detection_data
) {
	PROFILE_AUDIO_FUNCTION()

	std::ranges::copy(depth_estimation_data, this->depthEstimationData.begin());
	this->objectDetectionData = std::move(object_detection_data);

	processingFinished = false;
	if(!audio_settings.depth_audio_paused)
		processDepthEstimationData();
	if(!audio_settings.object_audio_paused)
		processObjectDetectionData();
	processingFinished = true;
}

void SpatialAudio::processDepthEstimationData() {
	PROFILE_AUDIO_FUNCTION()
	/*
	Processes the depth-estimation data:
	- dividing data into columns
	- getting nearest distance in column
	- calculate sound origin of nearest distance
	- saving this in AudioSource
	- updating audio sources in AudioMain
	Note: depending on the value of NUMBER_OF_SOURCES,
	not always all columns are used
	*/

	
	int step_size =
		SpatialAudioSettings::picture_x_resolution /
		audio_settings.NUMBER_OF_SOURCES; // NUMBER_OF_SOURCES = 2^x!
	CalculateSoundOrigin calculateSoundOrigin;
	std::vector<DepthAudioSourceData> new_audio_source_data;
	new_audio_source_data.reserve(SpatialAudioSettings::picture_x_resolution / step_size);

	LOG_INFO("[ProcessDepthEstimationData] Started processing...");

	for (int i = 0; i < SpatialAudioSettings::picture_x_resolution; i += step_size) {
		/*
		Because the data doesn't come in a 2d form, it is
		necessary to extract all elements of a column first.
		i + (j * row_length) represents all elements of a column.
		*/
		float nearest_distance = std::numeric_limits<float>::max();
		for (int j = 0; j < SpatialAudioSettings::picture_y_resolution; ++j) {
			float current_value = depthEstimationData[i + (j * SpatialAudioSettings::picture_x_resolution)];
			nearest_distance = std::min(current_value, nearest_distance);
		}

		/*
		Retrieving sound origin, by passing coordinates in Frame and nearest
		distance to the CalculateSoundOrigin::calculateSoundOrigin()
		function, which retruns the origin as a vector of coordinates
		*/
		std::array<float, 3> sound_origin =
			calculateSoundOrigin.calculateSoundOrigin(
				std::array<int, 2>{i + 1, 0}, nearest_distance,
				SpatialAudioSettings::picture_x_resolution
			);
		// float actual_distance = sqrt(sound_origin[0] * sound_origin[0] *
		// sound_origin[1] * sound_origin[1]);
		//float frequency = 250 - (100 * (std::fabs(calculateSoundOrigin.pixelAngle) / 90));
		new_audio_source_data.emplace_back(
				200, SpatialAudioSettings::BUFFER_DURATION,
				SpatialAudioSettings::SAMPLE_RATE, sound_origin[0], sound_origin[1],
				sound_origin[2]
			
		);
	}
	audio_main.changeDepthAudioData(new_audio_source_data);
	LOG_INFO("[ProcessDepthEstimationData] Finished processing...");
}

void SpatialAudio::processObjectDetectionData() {
	PROFILE_AUDIO_FUNCTION()
	/*
int column_length = SpatialAudioSettings::picture_y_resolution;	Processing object detection data:
	- going through all recognized objects
	- retrieving objects data
	- saving the data in the vector
	*/

	LOG_INFO("[ProcessObjectDetectionData] Started processing...");
	std::vector<ObjectAudioSourceData> new_audio_source_data;
	for (auto object : objectDetectionData) {
<<<<<<< HEAD
		std::string object_name = trim(toLower(object.cls_name));
=======
		const auto& box = object.bounding_box;

>>>>>>> main
		/*
		object coordinates are represented by values between 0 and 1
		so they need to be converted
		*/
		int x_coord =
			(int)(box.cx * (audio_settings.picture_x_resolution - 1));
		int y_coord =
<<<<<<< HEAD
			(int)(object.cy * (audio_settings.picture_y_resolution - 1));
	
		if(!object_label_data.contains(object_name)){
			LOG_ERROR(std::format("[ProcessObjectDetectionData] Could not find object {} in the object_label_data. Skipping to next one ...",object_name));
			continue;
		}
		int label_sound_start = object_label_data[object_name][0];
		int label_sound_end = object_label_data[object_name][1];
=======
			(int)(box.cy * (audio_settings.picture_y_resolution - 1));
		int label_sound_start = object_label_data[box.cls_name][0];
		int label_sound_end = object_label_data[box.cls_name][1];
>>>>>>> main

		// retrieving distance
		float distance = depthEstimationData.at(
			x_coord + (y_coord * audio_settings.picture_x_resolution)
		);

		std::array<float, 3> sound_origin =
			CalculateSoundOrigin().calculateSoundOrigin(
				std::array<int, 2>{x_coord + 1, 0}, distance,
				audio_settings.picture_x_resolution
			);
		new_audio_source_data.push_back(
			ObjectAudioSourceData{
				object_name,
				label_sound_start, label_sound_end, sound_origin[0],
				sound_origin[1], sound_origin[2]
			}
		);
	}
	audio_main.changeObjectAudioData(new_audio_source_data);
	LOG_INFO("[ProcessObjectDetectionData] Finished processing...");
}

void SpatialAudio::readObjectLabelData() {
	/*
	Reading the json file where the information about objects is stored:
	- name of object
	- start and end of the sound for the object in the big .wav file
	*/
	LOG_INFO("[ReadingObjectLabelData] Reading Object Label data...");

	std::string json_string(
		reinterpret_cast<const char*>(audio_settings.coco_labels_data.data()),
		audio_settings.coco_labels_data.size()
	);
	nlohmann::json json_object_data;
	try {
		json_object_data = nlohmann::json::parse(json_string);
	} catch (const nlohmann::json::parse_error& e) {
		LOG_ERROR(
			"[ReadingObjectLabelData] Could not parse JSON Data from Object "
			"Label data file"
		);
	}
	for (auto const& data : json_object_data) {
		object_label_data[toLower(data["text"])] = {data["start"], data["end"]};
	}
	for(const auto& [key, value] : object_label_data){
		LOG_INFO(std::format("[ReadingObjectLabelData] {}: Begin {}, End {}",key, value[0], value[1]));
	}
	LOG_INFO("[ReadingObjectLabelData] Finished reading Object Label data...");
}

std::string SpatialAudio::toLower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return result;
}

std::string SpatialAudio::trim(const std::string& str) {
    size_t start = 0;
    while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start]))) {
        ++start;
    }

    size_t end = str.size();
    while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1]))) {
        --end;
    }

    return str.substr(start, end - start);
}

bool SpatialAudio::getProcessingStatus() { return processingFinished; }

SpatialAudio::~SpatialAudio() {
	depth_audio_running = false;
	if (depth_audio_thread.joinable()) {
		depth_audio_thread.join();
	}
	object_audio_running = false;
	if (object_audio_thread.joinable()) {
		object_audio_thread.join();
	}
}
