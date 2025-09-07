#pragma once

#include <mutex>
#include <utility>

/// Helper class that encapsulates T value and protects every access to it using
/// a mutex
template<typename T>
class MutexGuard {
  public:
	explicit MutexGuard(T&& value) noexcept : value(std::move(value)) {}

	template<typename... Args>
	explicit MutexGuard(Args&&... args) noexcept
		: value(std::forward<Args...>(args)...) {}

	~MutexGuard() = default;

	MutexGuard(MutexGuard&&) noexcept = default;
	MutexGuard(const MutexGuard&) = delete;
	MutexGuard& operator=(MutexGuard&&) noexcept = default;
	MutexGuard& operator=(const MutexGuard&) = delete;

	struct ScopedAccess {
		explicit ScopedAccess(T& value, std::mutex& mutex)
			: value(value), lock(mutex) {}
		~ScopedAccess() = default;

		ScopedAccess(ScopedAccess&&) noexcept = default;
		ScopedAccess(const ScopedAccess&) = delete;
		ScopedAccess& operator=(ScopedAccess&&) noexcept = default;
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

	/// @return RAII object that locks the mutex and provides access to the
	/// value
	ScopedAccess lock() { return ScopedAccess(value, mutex); }

	struct ConstScopedAccess {
		explicit ConstScopedAccess(const T& value, std::mutex& mutex)
			: value(value), lock(mutex) {}
		~ConstScopedAccess() = default;

		ConstScopedAccess(ConstScopedAccess&&) noexcept = default;
		ConstScopedAccess(const ConstScopedAccess&) = delete;
		ConstScopedAccess& operator=(ConstScopedAccess&&) noexcept = default;
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

	/// @return RAII object that locks the mutex and provides access to the
	/// value
	ConstScopedAccess const_lock() const {
		return ConstScopedAccess(value, mutex);
	}

  private:
	T value;
	mutable std::mutex mutex;
};