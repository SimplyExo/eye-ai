#pragma once

#include <jni.h>
#include <span>
#include <string_view>

struct NativeFloatArrayScope {
	explicit NativeFloatArrayScope(JNIEnv* env, jfloatArray array);

	~NativeFloatArrayScope();

	NativeFloatArrayScope(const NativeFloatArrayScope&) = delete;
	NativeFloatArrayScope(NativeFloatArrayScope&&) = delete;
	void operator=(const NativeFloatArrayScope&) = delete;
	void operator=(NativeFloatArrayScope&&) = delete;

	[[nodiscard]] explicit(false) operator std::span<const float>() const {
		return native_array;
	}
	[[nodiscard]] explicit(false) operator std::span<float>() {
		return native_array;
	}

	[[nodiscard]] size_t size() const { return native_array.size(); }

  private:
	jfloatArray array = nullptr;
	JNIEnv* env = nullptr;
	std::span<jfloat> native_array;
};

struct NativeByteArrayScope {
	explicit NativeByteArrayScope(JNIEnv* env, jbyteArray array);

	~NativeByteArrayScope();

	NativeByteArrayScope(NativeByteArrayScope&&) = delete;
	NativeByteArrayScope(const NativeByteArrayScope&) = delete;
	void operator=(const NativeByteArrayScope&) = delete;
	void operator=(NativeByteArrayScope&&) = delete;

	[[nodiscard]] explicit(false) operator std::span<const jbyte>() const {
		return native_array;
	}
	[[nodiscard]] explicit(false) operator std::span<jbyte>() {
		return native_array;
	}

	[[nodiscard]] size_t size() const { return native_array.size(); }

  private:
	jbyteArray array = nullptr;
	JNIEnv* env = nullptr;
	std::span<jbyte> native_array;
};

struct NativeIntArrayScope {
	explicit NativeIntArrayScope(JNIEnv* env, jintArray array);

	~NativeIntArrayScope();

	NativeIntArrayScope(NativeIntArrayScope&&) = delete;
	NativeIntArrayScope(const NativeIntArrayScope&) = delete;
	void operator=(NativeIntArrayScope&&) = delete;
	void operator=(const NativeIntArrayScope&) = delete;

	[[nodiscard]] explicit(false) operator std::span<const jint>() const {
		return native_array;
	}
	[[nodiscard]] explicit(false) operator std::span<jint>() {
		return native_array;
	}

	[[nodiscard]] size_t size() const { return native_array.size(); }

  private:
	jintArray array = nullptr;
	JNIEnv* env = nullptr;
	std::span<jint> native_array;
};

struct NativeStringScope {
	explicit NativeStringScope(JNIEnv* env, jstring string);

	~NativeStringScope();

	NativeStringScope(NativeStringScope&&) = delete;
	NativeStringScope(const NativeStringScope&) = delete;
	void operator=(NativeStringScope&&) = delete;
	void operator=(const NativeStringScope&) = delete;

	[[nodiscard]] explicit(false) operator std::string_view() const {
		return native_string;
	}

  private:
	JNIEnv* env = nullptr;
	jstring string = nullptr;
	const char* native_string;
};