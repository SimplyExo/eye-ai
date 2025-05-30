#pragma once

#include "EyeAICore/utils/MutexGuard.hpp"
#include <AL/al.h>
#include <AL/alc.h>
#include <array>
#include <atomic>
#include <cmath>
#include <span>
#include <thread>
#include <vector>

struct OscillatorInfo {
	float amplitude = 0.5f;
	float frequency = 440.0f;
	/** left (-), right (+) */
	float position_x = 0.0f;
	/** bottom (-), top (+) */
	float position_y = 0.0f;
	/** front (-), back (+) */
	float position_z = 0.0f;
};

using AudioLogCallback = void (*)(std::string_view);

struct Oscillator {
	explicit Oscillator(const OscillatorInfo& info);
	~Oscillator() noexcept;

	Oscillator(Oscillator&&) = default;
	Oscillator(const Oscillator&) = delete;
	Oscillator& operator=(Oscillator&&) = delete;
	Oscillator& operator=(const Oscillator&) = delete;

	void play();
	void stop();

	void update_info(const OscillatorInfo& new_info);

	void fill_unqueued_buffer();

	constexpr static int SAMPLE_RATE = 44100;
	constexpr static int FORMAT = AL_FORMAT_MONO16; // short

  private:
	/** 4096/44100 = 0.0928798 seconds per buffer  -->  ~9ms latency */
	constexpr static int BUFFER_SIZE = 4096;
	constexpr static int NUM_BUFFERS = 4;
	constexpr static float TWO_PI = 2.0f * M_PI;

	void generate(std::span<short> buffer);

	OscillatorInfo info;
	ALuint source = 0;
	std::array<ALuint, NUM_BUFFERS> buffers{};
	float phase = 0.0f;
};

class SpatialAudioEngine {
  public:
	explicit SpatialAudioEngine(
		AudioLogCallback log_warning_callback,
		AudioLogCallback log_error_callback
	);
	~SpatialAudioEngine() noexcept;

	SpatialAudioEngine(SpatialAudioEngine&&) = delete;
	SpatialAudioEngine(const SpatialAudioEngine&) = delete;
	SpatialAudioEngine& operator=(SpatialAudioEngine&&) = delete;
	SpatialAudioEngine& operator=(const SpatialAudioEngine&) = delete;

	void enable();
	void disable() noexcept;

	void start();
	void stop();

	void update_oscillators(const std::vector<OscillatorInfo>& oscillator_infos
	);

  private:
	void streaming_thread();

	AudioLogCallback log_warning_callback;
	AudioLogCallback log_error_callback;

	bool enabled = false;
	ALCdevice* device = nullptr;
	ALCcontext* context = nullptr;

	// TODO: use triple buffering instead
	MutexGuard<std::vector<Oscillator>> oscillators;

	std::thread audio_streaming_thread;
	std::atomic_bool streaming = false;
};
