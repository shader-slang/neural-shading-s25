#!/bin/bash
# Setup script for Neural Shading Hardware Acceleration
# This script initializes the required Git submodules

set -e  # Exit on any error

echo "Setting up project dependencies..."

# Check if we're in a git repository
if ! git status &>/dev/null; then
    echo "Error: This is not a Git repository"
    echo "Please make sure you're in the correct directory"
    exit 1
fi

# Check if external directory exists
if [ ! -d "external" ]; then
    mkdir external
fi

# Add slang-rhi submodule if it doesn't exist
if [ ! -d "external/slang-rhi" ]; then
    echo "Adding slang-rhi submodule..."
    git submodule add https://github.com/shader-slang/slang-rhi.git external/slang-rhi
fi

# Add lz4 submodule if it doesn't exist
if [ ! -d "external/lz4" ]; then
    echo "Adding lz4 submodule..."
    git submodule add https://github.com/lz4/lz4.git external/lz4
fi

# Add miniz submodule if it doesn't exist
if [ ! -d "external/miniz" ]; then
    echo "Adding miniz submodule..."
    git submodule add https://github.com/richgel999/miniz.git external/miniz
fi

# Add unordered_dense submodule if it doesn't exist
if [ ! -d "external/unordered_dense" ]; then
    echo "Adding unordered_dense submodule..."
    git submodule add https://github.com/martinus/unordered_dense.git external/unordered_dense
fi

# Initialize and update all submodules
echo "Initializing and updating submodules..."
git submodule update --init --recursive

echo "Setup completed successfully!"
echo "You can now run build.sh to build the project."
