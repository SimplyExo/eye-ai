# EyeAICore

the core implementation of EyeAI in crossplatform c++

## Building EyeAICore from source using Docker container

Build docker image:

```bash
cd .. # Dockerfile is located at the root, since it copies MiDaS from EyeAIApp
docker build -t eye-ai .
```

Now you have a docker image that contains a build of EyeAICore from source.

To run the docker image in a container:

```bash
docker run -it eye-ai
```

## Building EyeAICore from source locally

Requirements:

- C++ compiler that supports all features of C++ 20. (for example: GCC 13+, Clang 14+ or MSVC 19.30+)
- CMake: version 3.22 or higher
- Git
- Python 3

Optional programs to install:

- Ninja: build system (used by the cmake presets, package is often named: ninja-build)
- clang-tidy: static code analysis (make sure to use at least clang-tidy-15, clang-tidy-14 has known issues with `<chrono>`)
- ccache: cache compilation results for faster repeated builds
- doxygen: generate documentation from source code

```bash
cmake --preset=release
cmake --build --preset=release
```

## Run EyeAICore Tests

```bash
cmake --preset=debug
cmake --build --preset=debug
ctest --preset=default
```

## Profile EyeAICore

- For EyeAIApp: select the "profiling" build type

- For running EyeAICore on your PC:

  ```bash
  cmake --preset=profiling
  cmake --build --preset=profiling
  ```

  Then run EyeAICore and listen with the Tracy profiler

## Generating EyeAICore docs using doxygen

install doxygen (here: using linux):

```bash
sudo apt install doxygen graphviz
```

generate docs:

```bash
cmake --preset=debug
cmake --build --preset=generate_docs
```

the generated doxygen docs will be in `build/docs`
