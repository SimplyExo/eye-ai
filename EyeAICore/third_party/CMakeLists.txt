
# LiteRT (TFLite)
set(LITERT_INCLUDE_DIRS
	"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/include/external/org_tensorflow"
)
set(LITERT_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/litert-gpu-1.2.0/include")
set(LITERT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/lib/${EYE_AI_CORE_ABI}/libtensorflowlite_jni.so")
set(LITERT_GPU_LIB "${CMAKE_CURRENT_SOURCE_DIR}/litert-gpu-1.2.0/lib/${EYE_AI_CORE_ABI}/libtensorflowlite_gpu_jni.so")

# OnnxRuntime
set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-android-1.22.0/include")
set(ONNXRUNTIME_LIB "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-android-1.22.0/lib/${EYE_AI_CORE_ABI}/libonnxruntime.so")


set(THIRD_PARTY_INCLUDE_DIRS
	${LITERT_INCLUDE_DIRS}
	${LITERT_GPU_INCLUDE_DIR}
	${ONNXRUNTIME_INCLUDE_DIR}
	PARENT_SCOPE
)
set(THIRD_PARTY_LIBS
	${LITERT_LIB}
	${LITERT_GPU_LIB}
	${ONNXRUNTIME_LIB}
	PARENT_SCOPE
)