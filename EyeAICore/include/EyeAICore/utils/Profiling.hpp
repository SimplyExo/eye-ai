#pragma once

#include <chrono>
#include <concurrentqueue.h>
#include <string_view>

using moodycamel::ConcurrentQueue;

using profile_clock = std::chrono::high_resolution_clock;

class ProfilingFrame;

struct ProfileScope {
	explicit ProfileScope(std::string_view name, ProfilingFrame& frame);
	~ProfileScope() noexcept;

	ProfileScope(const ProfileScope&) = delete;
	ProfileScope(ProfileScope&&) = delete;
	void operator=(const ProfileScope&) = delete;
	void operator=(ProfileScope&&) = delete;

  private:
	std::string_view name;
	int scope_depth = 0;
	ProfilingFrame& frame;
	profile_clock::time_point start;
};

struct ProfileScopeRecord {
	std::string_view name;
	int scope_depth = 0;
	profile_clock::time_point start;
	profile_clock::duration duration;

	[[nodiscard]] std::string formatted() const;
};

/// collection of profile records from different threads (lock-free thread-safe)
class ProfilingFrame {
  public:
	explicit ProfilingFrame(std::string_view name) : name(name) {}

	/// returns the scopes depth, should always include calling end_scope after
	int start_scope() noexcept;

	void end_scope(const ProfileScopeRecord& scope) noexcept;

	/// returns formatted info of the finished frame and clears all contents to
	/// start a new frame
	std::string finish();

  private:
	std::string_view name;
	profile_clock::time_point start = profile_clock::now();
	ConcurrentQueue<ProfileScopeRecord> profile_scopes;
	std::atomic_int current_frame_scope_depth = 0;
};

/// These four functions return global static variables (needed since NativeLib
/// is loaded as a shared library, so a simple static variable does not work).
/// Both ProfilingFrame's are thread-safe

ProfilingFrame& get_depth_profiling_frame();
void set_last_depth_profiling_frame_formatted(std::string&& formatted);
std::string get_last_depth_profiling_frame_formatted();
ProfilingFrame& get_camera_profiling_frame();
void set_last_camera_profiling_frame_formatted(std::string&& formatted);
std::string get_last_camera_profiling_frame_formatted();

#define COMBINE(x, y) x##y
#define COMBINE2(x, y) COMBINE(x, y)
#define PROFILE_DEPTH_SCOPE(name)                                              \
	const ProfileScope COMBINE2(__profile_scope_, __LINE__)(                   \
		name, get_depth_profiling_frame()                                      \
	);
#define PROFILE_CAMERA_SCOPE(name)                                             \
	const ProfileScope COMBINE2(__profile_scope_, __LINE__)(                   \
		name, get_camera_profiling_frame()                                     \
	);

#ifndef FUNCTION_NAME
#ifdef WIN32 // WINDOWS
#define FUNCTION_NAME() __FUNCTION__
#else //*NIX
#define FUNCTION_NAME() __func__
#endif
#endif

#define PROFILE_DEPTH_FUNCTION()                                               \
	const ProfileScope COMBINE2(__profile_scope_, __LINE__)(                   \
		FUNCTION_NAME(), get_depth_profiling_frame()                           \
	);

#define PROFILE_CAMERA_FUNCTION()                                              \
	const ProfileScope COMBINE2(__profile_scope_, __LINE__)(                   \
		FUNCTION_NAME(), get_camera_profiling_frame()                          \
	);
