#pragma once

class AudioSettings {
  public:
	float BUFFER_DURATION = 1.0f;
	int BUFFERS_PER_SOURCE = 3;
	int SAMPLE_RATE = 4800;
	int NUMBER_OF_SOURCES;
	float FREQUENCY;

	AudioSettings(
		int num_of_sources = 8,
		float freq = 150.0f
	)
		: NUMBER_OF_SOURCES(num_of_sources),
		  FREQUENCY(freq) {}
};

extern AudioSettings audio_settings;