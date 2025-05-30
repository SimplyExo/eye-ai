#pragma once

#include <mutex>
#include <utility>

/** Helper class that encapsulates T value and protects every access to it using
 * a mutex */
template<typename T>
class MutexGuard {
  public:
	template<typename... Args>
	explicit MutexGuard(Args&&... args)
		: value(std::forward<Args...>(args)...) {}

	~MutexGuard() = default;

	MutexGuard(MutexGuard&&) = default;
	MutexGuard(const MutexGuard&) = default;
	MutexGuard& operator=(MutexGuard&&) = default;
	MutexGuard& operator=(const MutexGuard&) = default;

	struct ScopedAccess {
		explicit ScopedAccess(T& value, std::mutex& mutex)
			: value(value), lock(mutex) {}
		~ScopedAccess() = default;

		ScopedAccess(ScopedAccess&&) = default;
		ScopedAccess(const ScopedAccess&) = delete;
		ScopedAccess& operator=(ScopedAccess&&) = default;
		ScopedAccess& operator=(const ScopedAccess&) = delete;

		T* operator->() { return &value; }
		T& operator*() { return value; }

		template<typename Index>
		auto& operator[](Index&& index) {
			return value[std::forward<Index>(index)];
		}

	  private:
		T& value;
		std::lock_guard<std::mutex> lock;
	};

	ScopedAccess lock() { return ScopedAccess(value, mutex); }

	struct ConstScopedAccess {
		explicit ConstScopedAccess(const T& value, std::mutex& mutex)
			: value(value), lock(mutex) {}
		~ConstScopedAccess() = delete;

		ConstScopedAccess(ConstScopedAccess&&) = default;
		ConstScopedAccess(const ConstScopedAccess&) = delete;
		ConstScopedAccess& operator=(ConstScopedAccess&&) = default;
		ConstScopedAccess& operator=(const ConstScopedAccess&) = delete;

		const T* operator->() { return &value; }
		const T& operator*() { return value; }

		template<typename Index>
		const auto& operator[](Index&& index) {
			return value[std::forward<Index>(index)];
		}

	  private:
		const T& value;
		std::lock_guard<std::mutex> lock;
	};

	ConstScopedAccess lock() const { return ConstScopedAccess(value, mutex); }

  private:
	T value;
	mutable std::mutex mutex;
};