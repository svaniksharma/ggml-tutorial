# Simple GGML Tutorial

A basic introduction to GGML. Accompanies the article ["A Short Introduction to GGML"](https://svaniksharma.github.io/posts/2025-05-28-a-short-ggml-tutorial/).

## Build Instructions for Executable

Run the following if you want to build with CUDA:

```bash
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
make
./tutorial
```

If you don't want/don't have CUDA, then run the following for CPU:

```bash
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
make
./tutorial
```

## Use WebAssembly

To run (part of) the tutorial in WASM, you will need [Emscripten](https://emscripten.org/). How to run:

```bash
mkdir build
cd build
emcmake cmake ..
emmake make
```

Then, copy `tutorial.js` into the `web` directory. You will need Python 3 to execute `server.py`:

```bash
cp tutorial.js ../web
cd ../web
python server.py
```

Then, go to the address printed out on the terminal. Open up the in-browser console. You should see the output:

```shell
Tutorial Result: 19
Using CPU as backend
Backend result: 19
```
