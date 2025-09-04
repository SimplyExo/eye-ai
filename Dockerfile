FROM docker.io/gcc:13.1

RUN apt-get update && \
	apt-get install -y cmake && \
	apt-get install -y python3 && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /eye-ai

# EyeAICore code
COPY ./EyeAICore ./EyeAICore

# MiDaS tflite model (from EyeAIApp)
COPY ./EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite ./EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite

RUN cd EyeAICore && cmake -B build -DENABLE_CLANG_TIDY=OFF && cmake --build build -j8
