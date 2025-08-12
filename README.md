# Neural Shading SIGGRAPH 2025

Materials for the Neural Shading Course at SIGGRAPH 2025. This repository contains practical examples and implementations of neural shading techniques using Slang.

## Overview

This course covers the fundamentals of neural shading using Slang. The materials include both Python-based examples using `slangpy` and C++ implementations for high-performance neural network training on GPU.

## Directory Structure

```
├── autodiff/                             # Automatic differentiation examples
│   ├── README.md                         # Autodiff examples documentation
│   ├── square.slang                      # Basic square function
│   ├── square-struct.slang               # Structured automatic differentiation
│   └── square-debug.slang                # Debugging automatic differentiation
│
├── hardware-acceleration/                # High-performance GPU implementations
│   ├── mlp-training/                     # Basic MLP training implementation
│   ├── mlp-training-coopvec/             # Cooperative vector MLP training (only supported on Nvidia hardware on Windows/Linux)
│   ├── example-base/                     # Shared base library
│   └── external/                         # External dependencies (slang-rhi, etc.)
│
├── mipmap/                               # Mipmap and texture filtering examples
│   ├── app.py                            # Main application framework
│   ├── app.slang                         # Main application shader
│   ├── brdf.slang                        # BRDF shader implementation
│   ├── step_01_basicprogram.py           # Basic shader program
│   ├── step_01_basicprogram.slang        # Basic shader program implementation
│   ├── step_02_mipmap.py                 # Mipmap visualisation
│   ├── step_02_mipmap.slang              # Mipmap shader implementation
│   ├── step_03_supersample.py            # Supersampling visualisation
│   ├── step_04_loss.py                   # Loss function visualisation
│   ├── step_04_loss.slang                # Loss function shader implementation
│   ├── step_05_train.py                  # Training the mipmap using the loss function
│   ├── step_05_train.slang               # Training shader implementation
│   ├── PavingStones070_2K.diffuse.jpg    # Texture files
│   ├── PavingStones070_2K.normal.jpg
│   └── PavingStones070_2K.roughness.jpg
│
└── network/                              # Neural network examples
    ├── app.py                            # Main application framework
    ├── app.slang                         # Main application shader
    ├── step_01_basicnetwork.py           # Basic single-layer neural network
    ├── step_01_basicnetwork.slang        # Basic network shader implementation
    ├── step_02_multiple_layers.py        # Multi-layer network example
    ├── step_02_multiple_layers.slang     # Multi-layer network shader
    ├── step_03_better_activations.py     # Network with improved activations
    ├── step_03_better_activations.slang  # Improved activations shader
    ├── step_04_frequency_encoding.py     # Frequency encoding example
    ├── step_04_frequency_encoding.slang  # Frequency encoding shader
    ├── step_05_latent_texture.py         # Latent texture training example
    ├── step_05_latent_texture.slang      # Latent texture shader
    └── slangstars.png                    # Example texture
```

## Prerequisites

### For Python Examples (mipmap/, network/)
- Python 3.7 or later
- `slangpy` - Python bindings for Slang
- `numpy` - Numerical computing
- `pillow` - Image processing
- `slang` - latest Slang release from https://github.com/shader-slang/slang/releases for autodiff

### For C++ Examples (hardware-acceleration/)
- CMake 3.20 or later
- C++17 compatible compiler
- Graphics driver supporting Vulkan (or Metal on macOS, note that cooperative vector is not supported on Metal)
- Git

## Installation

### Python Environment Setup

1. **Install slangpy**:
   ```bash
   pip install slangpy
   ```

2. **Install additional dependencies**:
   ```bash
   pip install numpy pillow
   ```

### C++ Environment Setup

1. **Initialize submodules**:
   ```bash
   cd hardware-acceleration
   git submodule update --init --recursive
   ```

2. **Build the hardware-acceleration project**:
   ```bash
   # On Windows
   cd hardware-acceleration
   setup.bat
   build.bat
   
   # On Linux/macOS
   cd hardware-acceleration
   ./setup.sh
   ./build.sh
   ```

3. **Download Slang to test the autodiff shaders**:
- Download the latest Slang release from https://github.com/shader-slang/slang/releases
- Use the bin/slangi binary to compile and run the autodiff shader files.

## How to Run

### Running Python Examples

**Basic Mipmap Example** (neural-shading-s25/mipmap):
```bash
cd mipmap
python step_01_basicprogram.py
```

**Neural Network Training** (neural-shading-s25/mipmap):
```bash
cd mipmap
python step_05_train.py
```

**Basic Network Example** (neural-shading-s25/network):
```bash
cd network
python step_01_basicnetwork.py
```

### Running C++ Examples

**Autodiff Examples** (neural-shading-s25/autodiff):
```bash
cd autodiff
./slangi square.slang
```

**Basic MLP Training** (neural-shading-s25/hardware-acceleration):
```bash
cd hardware-acceleration/build/Release  # or build/ on Linux/macOS
./mlp-training
```

**Cooperative Vector MLP Training** (neural-shading-s25/hardware-acceleration):
```bash
cd hardware-acceleration/build/Release  # or build/ on Linux, this is not supported on macOS
./mlp-training-coopvec
```

## Key Features

### Automatic Differentiation
- Examples demonstrating Slang's automatic differentiation capabilities
- Basic function differentiation with `bwd_diff`
- Structured differentiation for complex types

### Hardware Acceleration
- GPU-accelerated neural network training
- Cooperative vector intrinsics for improved performance (only on Nvidia hardware on Windows and Linux)
- Cross-platform support (Vulkan, Metal)

### Neural Shading
- Real-time neural network inference in shaders
- Texture-based neural networks
- Training and inference pipelines

### Mipmap Techniques
- Advanced texture filtering
- Neural mipmap generation
- Loss function implementations

## Platform Support

General Platform Support:
- **Windows**: Vulkan
- **Linux**: Vulkan
- **macOS**: Metal (except hardware acceleration examples)

Hardware Acceleration Examples:
- **mlp-training**: Windows, Linux, and macOS (uses Vulkan on Windows/Linux, Metal on macOS)
- **mlp-training-coopvec**: Windows and Linux with NVIDIA GPU only (cooperative vector not supported on macOS)

## Troubleshooting

### Common Issues

1. **slangpy not found**: Ensure you have installed slangpy via pip
2. **Build failures**: Check that all submodules are initialized
3. **Graphics driver issues**: Update to latest graphics drivers
4. **mlp-training-coopvec issues**: Cooperative vector is only supported on Nvidia hardware, on Windows and Linux

### Getting Help

- Check the individual README files in each subdirectory
- Review the hardware-acceleration README for detailed build instructions
- Ensure all dependencies are properly installed

## Useful Resources

### RTXNS

The RTX neural shading SDK is a useful resource for developers interested in bringing machine learning to graphics applications.
It provides examples for training neural networks and using the models to perform inferences on normal graphics rendering.
https://github.com/NVIDIA-RTX/RTXNS

### RTXNC

Related to the neural shaing samples is the RTX Neural Texture Compression (NTC) SDK which allows compressing all
PBR textures for a single material together: https://github.com/NVIDIA-RTX/RTXNTC

## Contributing

This repository contains course materials for SIGGRAPH 2025. For questions or issues related to the course content, please refer to the course documentation or contact the instructors.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- Slang team for the shader language and compiler
- slang-rhi and slangpy team and contributors
- SIGGRAPH 2025 Neural Shading Course instructors and contributors
