# EyeAICore

the core implementation of EyeAI in crossplatform c++

## Generating EyeAICore docs using doxygen

install doxygen (here: using linux):

```bash
sudo apt install doxygen graphviz
```

generate docs:

```bash
cmake -B build -DGENERATE_DOCS=ON
cmake --build build --target EyeAICoreDoxygenDocs
```

the generated doxygen docs will be in `build/docs`
