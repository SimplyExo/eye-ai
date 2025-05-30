#include "NativeJavaScopes.hpp"

NativeFloatArrayScope::NativeFloatArrayScope(JNIEnv* env, jfloatArray array)
	: array(array), env(env) {
	const size_t length = env->GetArrayLength(array);
	jfloat* pointer = env->GetFloatArrayElements(array, nullptr);
	native_array = std::span<jfloat>(pointer, length);
}

NativeFloatArrayScope::~NativeFloatArrayScope() {
	env->ReleaseFloatArrayElements(array, native_array.data(), 0);
}

NativeByteArrayScope::NativeByteArrayScope(JNIEnv* env, jbyteArray array)
	: array(array), env(env) {
	const size_t length = env->GetArrayLength(array);
	jbyte* pointer = env->GetByteArrayElements(array, nullptr);
	native_array = std::span<jbyte>(pointer, length);
}

NativeByteArrayScope::~NativeByteArrayScope() {
	env->ReleaseByteArrayElements(array, native_array.data(), 0);
}

NativeIntArrayScope::NativeIntArrayScope(JNIEnv* env, jintArray array)
	: array(array), env(env) {
	const size_t length = env->GetArrayLength(array);
	jint* pointer = env->GetIntArrayElements(array, nullptr);
	native_array = std::span<jint>(pointer, length);
}

NativeIntArrayScope::~NativeIntArrayScope() {
	env->ReleaseIntArrayElements(array, native_array.data(), 0);
}

NativeStringScope::NativeStringScope(JNIEnv* env, jstring string)
	: string(string), env(env),
	  native_string(env->GetStringUTFChars(string, nullptr)) {}

NativeStringScope::~NativeStringScope() {
	env->ReleaseStringUTFChars(string, native_string);
}