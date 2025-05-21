#pragma once

#include <jni.h>
#include <span>
#include <string_view>

struct NativeFloatArrayScope {
	explicit NativeFloatArrayScope(JNIEnv* env, jfloatArray array)
		: array(array), env(env) {
		const size_t length = env->GetArrayLength(array);
		jfloat* pointer = env->GetFloatArrayElements(array, nullptr);
		native_array = std::span<jfloat>(pointer, length);
	}

	~NativeFloatArrayScope() {
		env->ReleaseFloatArrayElements(array, native_array.data(), 0);
	}

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
	explicit NativeByteArrayScope(JNIEnv* env, jbyteArray array)
		: array(array), env(env) {
		const size_t length = env->GetArrayLength(array);
		jbyte* pointer = env->GetByteArrayElements(array, nullptr);
		native_array = std::span<jbyte>(pointer, length);
	}

	~NativeByteArrayScope() {
		env->ReleaseByteArrayElements(array, native_array.data(), 0);
	}

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
	explicit NativeIntArrayScope(JNIEnv* env, jintArray array)
		: array(array), env(env) {
		const size_t length = env->GetArrayLength(array);
		jint* pointer = env->GetIntArrayElements(array, nullptr);
		native_array = std::span<jint>(pointer, length);
	}

	~NativeIntArrayScope() {
		env->ReleaseIntArrayElements(array, native_array.data(), 0);
	}

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
	explicit NativeStringScope(JNIEnv* env, jstring string)
		: string(string), env(env),
		  native_string(env->GetStringUTFChars(string, nullptr)) {}

	~NativeStringScope() { env->ReleaseStringUTFChars(string, native_string); }

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