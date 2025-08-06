#!/bin/bash
# Build script for Neural Shading Hardware Acceleration

set -e  # Exit on any error

BUILD_DIR="build"
CMAKE_GENERATOR="Unix Makefiles"

echo "Building Neural Shading Hardware Acceleration..."

# Check if slang-rhi submodule is initialized
if [ ! -f "external/slang-rhi/CMakeLists.txt" ]; then
    echo "Error: slang-rhi submodule not found or not initialized"
    echo "Please run setup.sh first to initialize submodules"
    exit 1
fi

# Check if lz4 submodule is initialized
if [ ! -f "external/lz4/build/cmake/CMakeLists.txt" ]; then
    echo "Error: lz4 submodule not found or not initialized"
    echo "Please run setup.sh first to initialize submodules"
    exit 1
fi

# Check if miniz submodule is initialized
if [ ! -f "external/miniz/CMakeLists.txt" ]; then
    echo "Error: miniz submodule not found or not initialized"
    echo "Please run setup.sh first to initialize submodules"
    exit 1
fi

# Check if unordered_dense submodule is initialized
if [ ! -f "external/unordered_dense/CMakeLists.txt" ]; then
    echo "Error: unordered_dense submodule not found or not initialized"
    echo "Please run setup.sh first to initialize submodules"
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

# Configure the project
echo "Configuring project..."
cmake -B "$BUILD_DIR" -G "$CMAKE_GENERATOR" -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
cmake --build "$BUILD_DIR" --config Release -j$(nproc)

echo "Build completed successfully!"
echo "Executables can be found in: $BUILD_DIR/"
