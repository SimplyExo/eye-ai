#include "SpatialAudioEngine.hpp"
#include "al.h"
#include "utils/Log.hpp"

SpatialAudioEngine::SpatialAudioEngine() { enable(); }

SpatialAudioEngine::~SpatialAudioEngine() noexcept { disable(); }

void SpatialAudioEngine::enable() {
	if (enabled) {
		LOG_WARN("SpatialAudioEngine::enable called, but already enabled");
		return;
	}

	device = alcOpenDevice(nullptr);
	if (device == nullptr) {
		LOG_ERROR("Failed to open audio device");
		return;
	}
	context = alcCreateContext(device, nullptr);
	if (context == nullptr) {
		LOG_ERROR("Failed to create audio context");
		return;
	}
	alcMakeContextCurrent(context);

	enabled = true;
}

void SpatialAudioEngine::disable() noexcept {
	if (!enabled) {
		LOG_WARN("SpatialAudioEngine::disable called, but already disabled");
		return;
	}

	oscillators.lock()->clear();

	alcMakeContextCurrent(nullptr);
	if (context != nullptr)
		alcDestroyContext(context);
	if (device != nullptr)
		alcCloseDevice(device);

	enabled = false;
}

void SpatialAudioEngine::start() {
	streaming = true;

	for (auto& oscillator : *oscillators.lock())
		oscillator.play();

	audio_streaming_thread = std::thread([this]() { streaming_thread(); });
	audio_streaming_thread.detach();
}

void SpatialAudioEngine::stop() {
	for (auto& oscillator : *oscillators.lock())
		oscillator.stop();

	streaming = false;
}

void SpatialAudioEngine::update_oscillators(
	const std::vector<OscillatorInfo>& oscillator_infos
) {
	auto oscillators_ref = oscillators.lock();

	oscillators_ref->reserve(oscillator_infos.size());
	for (size_t i = 0; i < oscillators_ref->size(); i++) {
		oscillators_ref[i].update_info(oscillator_infos[i]);
	}
	for (size_t i = oscillators_ref->size(); i < oscillators_ref->capacity();
		 i++) {
		auto& oscillator = oscillators_ref->emplace_back(oscillator_infos[i]);
		oscillator.play();
	}
}

void SpatialAudioEngine::streaming_thread() {
	while (streaming) {
		for (auto& oscillator : *oscillators.lock())
			oscillator.fill_unqueued_buffer();
	}
}

Oscillator::Oscillator(const OscillatorInfo& info) : info(info) {
	alGenSources(1, &source);
	alGenBuffers((ALsizei)buffers.size(), buffers.data());
	alSource3f(
		source, AL_POSITION, info.position_x, info.position_y, info.position_z
	);
}

Oscillator::~Oscillator() noexcept {
	stop();
	alDeleteSources(1, &source);

	ALint buffers_queued = 0;
	alGetSourcei(source, AL_BUFFERS_QUEUED, &buffers_queued);
	if (buffers_queued > 0) {
		auto unqueued_buffers = std::vector<ALuint>(buffers_queued);
		alSourceUnqueueBuffers(source, buffers_queued, unqueued_buffers.data());
	}
	alDeleteBuffers((ALsizei)buffers.size(), buffers.data());
}

void Oscillator::play() {
	std::array<short, BUFFER_SIZE> samples{};
	for (const auto buffer : buffers) {
		generate(std::span<short>(samples));
		const auto samples_data = std::as_bytes(std::span<short>(samples));
		alBufferData(
			buffer, AL_FORMAT_MONO16, samples_data.data(),
			(ALsizei)samples_data.size_bytes(), SAMPLE_RATE
		);
	}
	alSourceQueueBuffers(source, NUM_BUFFERS, buffers.data());

	alSourcePlay(source);
}

// NOLINTBEGIN(readability-make-member-function-const)
void Oscillator::stop() { alSourceStop(source); }
// NOLINTEND(readability-make-member-function-const)

void Oscillator::update_info(const OscillatorInfo& new_info) {
	info = new_info;
	alSource3f(
		source, AL_POSITION, info.position_x, info.position_y, info.position_z
	);
}

void Oscillator::fill_unqueued_buffer() {
	ALint processed = 0;
	alGetSourcei(source, AL_BUFFERS_PROCESSED, &processed);

	if (processed > 0) {
		ALuint unqueued_buffer = 0;
		alSourceUnqueueBuffers(source, 1, &unqueued_buffer);

		std::array<short, BUFFER_SIZE> samples{};
		generate(samples);
		const auto samples_data = std::as_bytes(std::span<short>(samples));
		alBufferData(
			unqueued_buffer, FORMAT, samples_data.data(),
			(ALsizei)samples_data.size_bytes(), SAMPLE_RATE
		);

		alSourceQueueBuffers(source, 1, &unqueued_buffer);
	}

	ALint state = 0;
	alGetSourcei(source, AL_SOURCE_STATE, &state);
	if (state != AL_PLAYING)
		alSourcePlay(source);
}

void Oscillator::generate(std::span<short> buffer) {
	const float phase_increment =
		(TWO_PI * info.frequency) / (float)SAMPLE_RATE;
	for (auto& sample : buffer) {
		if (phase < M_PI) {
			sample = static_cast<short>((float)SHRT_MAX * info.amplitude);
		} else {
			sample =
				static_cast<short>((float)SHRT_MAX * info.amplitude * -1.0f);
		}
		phase += phase_increment;
		if (phase >= TWO_PI)
			phase -= TWO_PI;
	}
}