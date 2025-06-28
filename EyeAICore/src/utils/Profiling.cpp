#include "EyeAICore/utils/Profiling.hpp"
#include "EyeAICore/utils/MutexGuard.hpp"
#include <algorithm>
#include <chrono>
#include <format>

static std::string padding_tabs(size_t amount) {
	std::string result;
	result.reserve(amount * 4);
	for (size_t i = 0; i < amount; i++) {
		result.append("    ");
	}
	return result;
}

static std::string format_duration_millis(profile_clock::duration duration) {
	const auto duration_micros =
		std::chrono::duration_cast<std::chrono::microseconds>(duration);
	return std::format(
		"{:.2f} ms", static_cast<float>(duration_micros.count()) / 1000.0f
	);
}

ProfileScope::ProfileScope(std::string_view name, ProfilingFrame& frame)
	: name(name), scope_depth(frame.start_scope()), frame(frame),
	  start(profile_clock::now()) {}

ProfileScope::~ProfileScope() noexcept {
	const auto duration = profile_clock::now() - start;
	frame.end_scope(ProfileScopeRecord(name, scope_depth, start, duration));
}

std::string ProfileScopeRecord::formatted() const {
	auto padding = padding_tabs(std::max(scope_depth, 0));
	return std::format(
		"{}{}: {}", padding, name, format_duration_millis(duration)
	);
}

int ProfilingFrame::start_scope() noexcept {
	return current_frame_scope_depth++;
}

void ProfilingFrame::end_scope(const ProfileScopeRecord& scope) noexcept {
	profile_scopes.enqueue(scope);
	current_frame_scope_depth--;
}

std::string ProfilingFrame::finish() {
	const auto end = profile_clock::now();

	std::vector<ProfileScopeRecord> profile_scopes_vector;
	profile_scopes_vector.reserve(profile_scopes.size_approx());
	ProfileScopeRecord tmp;
	while (profile_scopes.try_dequeue(tmp))
		profile_scopes_vector.push_back(tmp);

	std::ranges::sort(
		profile_scopes_vector,
		[](const auto& a, const auto& b) -> bool { return a.start < b.start; }
	);
	std::string profile_scopes_formatted;
	for (const auto& profile_scope : profile_scopes_vector) {
		profile_scopes_formatted +=
			std::format("    {}\n", profile_scope.formatted());
	}
	const auto frame_duration = end - start;
	const auto frame_duration_ms =
		static_cast<float>(
			std::chrono::duration_cast<std::chrono::microseconds>(
				frame_duration
			)
				.count()
		) /
		1000.0f;
	auto frame_fps = 1.0 / (frame_duration_ms / 1000.0f);
	auto formatted = std::format(
		"{} Frame: {:.2f} fps ({:.2f} ms)\n{}", name, frame_fps,
		frame_duration_ms, profile_scopes_formatted
	);

	current_frame_scope_depth = 0;
	start = profile_clock::now();

	return formatted;
}

// All 4 global variables are thread-safe.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static auto depth_profiling_frame = ProfilingFrame("Depth");
static MutexGuard<std::string> last_depth_profiling_frame_formatted;
static auto camera_profiling_frame = ProfilingFrame("Camera");
static MutexGuard<std::string> last_camera_profiling_frame_formatted;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

void set_last_depth_profiling_frame_formatted(std::string&& formatted) {
	*last_depth_profiling_frame_formatted.lock() = std::move(formatted);
}
std::string get_last_depth_profiling_frame_formatted() {
	return *last_depth_profiling_frame_formatted.lock();
}
ProfilingFrame& get_depth_profiling_frame() { return depth_profiling_frame; }

ProfilingFrame& get_camera_profiling_frame() { return camera_profiling_frame; }
void set_last_camera_profiling_frame_formatted(std::string&& formatted) {
	*last_camera_profiling_frame_formatted.lock() = std::move(formatted);
}
std::string get_last_camera_profiling_frame_formatted() {
	return *last_camera_profiling_frame_formatted.lock();
}