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
