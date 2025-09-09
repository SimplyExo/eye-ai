#pragma once

/*
Struct of all the data a source needs to play a specific object
at a specific location
*/

struct ObjectAudioSourceData {
	// time where the sound in the .wav file begins and where it ends (in ms)
	int sound_begin; 
	int sound_end;	 
	float x1_position;
	float x2_position;
	float x3_position;

	ObjectAudioSourceData(
		int sound_begin,
		int sound_end,
		float x1,
		float x2,
		float x3
	)
		: sound_begin(sound_begin), sound_end(sound_end), x1_position(x1),
		  x2_position(x2), x3_position(x3) {}
};