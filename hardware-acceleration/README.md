# Neural Shading Hardware Acceleration

This folder contains the source code used in the "Hardware Acceleration" section of the "Introduction to Neural Shading" cource at
SIGGRPAH 2025. This project includes two implementations of neural network (MLP) training using Slang and slang-rhi.

`mlp-trainining/` contains the shader and host-side C++ code to train an MLP without using any special intrinsics.
`mlp-training-coopvec/` is a modified version of `mlp-trainining` that uses cooperative vectors to speed up inferencing and training.

## Build Requirements

- CMake 3.20 or later
- C++17 compatible compiler (Visual Studio 2019+ on Windows, GCC 7+ or Clang 5+ on Linux)
- Internet connection for downloading external dependencies

## External Dependencies

The build system uses the following external libraries:
- [Slang](https://github.com/shader-slang/slang) - Shader language and compiler (automatically downloaded)
- [slang-rhi](https://github.com/shader-slang/slang-rhi) - Render hardware interface (Git submodule)
- [LZ4](https://github.com/lz4/lz4) - Fast compression library (Git submodule)
- [miniz](https://github.com/richgel999/miniz) - Compression library (Git submodule)
- [unordered_dense](https://github.com/martinus/unordered_dense) - Fast hash map/set library (Git submodule)

### Setting up submodules

Before building, you need to initialize the required submodules:

```bash
git submodule update --init --recursive
```

## Building

### Quick Start

1. **Setup dependencies** (first time only):
   ```cmd
   setup.bat
   ```

2. **Build the project**:
   ```cmd
   build.bat
   ```

### Manual Setup

If you prefer to set up manually:

1. **Initialize submodules**:
   ```bash
   git submodule add https://github.com/shader-slang/slang-rhi.git external/slang-rhi
   git submodule update --init --recursive
   ```

2. **Build with Visual Studio**:
   ```cmd
   mkdir build
   cmake -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release
   ```

### Cross-platform (Make/Ninja)

```bash
# Setup submodules (first time only)
git submodule update --init --recursive

# Build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Executables

The build produces two executables:

1. **mlp-training.exe** - Basic MLP training implementation
2. **mlp-training-coopvec.exe** - MLP training with cooperative vector intrinsics

Both executables implement a simple multi-layer perceptron trained to approximate polynomial expressions.

## Project Structure

```
├── CMakeLists.txt          # Main build configuration
├── build.bat               # Windows build script
├── example-base/           # Shared base library
│   ├── core/              # Core utilities and data structures
│   └── platform/          # Platform-specific code
├── mlp-training/          # Basic MLP training implementation
│   ├── *.slang           # Shader files
│   └── mlp-training.cpp  # Main implementation
└── mlp-training-coopvec/  # Cooperative vector MLP implementation
    ├── *.slang           # Shader files
    └── mlp-training-coopvec.cpp # Main implementation
```

## Running

After building, the executables will be located in:
- Windows: `build/Release/`
- Linux/macOS: `build/`

The shader (.slang) files are automatically copied to the build directory for runtime access.

## Troubleshooting

### Download Issues
If external dependencies fail to download:
1. Check your internet connection
2. Verify the GitHub repositories are accessible
3. Manually adjust version numbers in CMakeLists.txt if needed

### Build Issues
- Ensure you have all required build tools installed
- Check that your compiler supports C++17
- On Windows, make sure you have the Windows SDK installed

### Runtime Issues
- Ensure shader files (.slang) are in the same directory as the executable
- Check that your graphics drivers support Vulkan
- To run mlp-training-coopvec, make sure you have an NVIDIA GPU with latest driver.
