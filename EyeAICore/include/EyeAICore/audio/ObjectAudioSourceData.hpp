#pragma once

#include <iostream>
#include <utility>
/*
Struct of all the data a source needs to play a specific object
at a specific location
*/

struct ObjectAudioSourceData {
	std::string name;
	// time where the sound in the .wav file begins and where it ends (in ms)
	int sound_begin; 
	int sound_end;	 
	float x1_position;
	float x2_position;
	float x3_position;

	ObjectAudioSourceData(
		std::string name = "",
		int sound_begin = 0,
		int sound_end = 0,
		float x1 = 0.0f,
		float x2 = 0.0f,
		float x3 = 0.0f
	)
		: name(std::move(name)),sound_begin(sound_begin), sound_end(sound_end), x1_position(x1),
		  x2_position(x2), x3_position(x3) {}
};