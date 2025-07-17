# EyeAICore

the core implementation of EyeAI in crossplatform c++

## Building EyeAICore

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
