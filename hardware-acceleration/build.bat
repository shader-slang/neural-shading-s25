@echo off
REM Build script for Neural Shading Hardware Acceleration

set BUILD_DIR=build
set CMAKE_GENERATOR="Visual Studio 17 2022"

REM Check if slang-rhi submodule is initialized
if not exist external\slang-rhi\CMakeLists.txt (
    echo Error: slang-rhi submodule not found or not initialized
    echo Please run setup.bat first to initialize submodules
    pause
    exit /b 1
)

REM Check if lz4 submodule is initialized
if not exist external\lz4\build\cmake\CMakeLists.txt (
    echo Error: lz4 submodule not found or not initialized
    echo Please run setup.bat first to initialize submodules
    pause
    exit /b 1
)

REM Check if miniz submodule is initialized
if not exist external\miniz\CMakeLists.txt (
    echo Error: miniz submodule not found or not initialized
    echo Please run setup.bat first to initialize submodules
    pause
    exit /b 1
)

REM Check if unordered_dense submodule is initialized
if not exist external\unordered_dense\CMakeLists.txt (
    echo Error: unordered_dense submodule not found or not initialized
    echo Please run setup.bat first to initialize submodules
    pause
    exit /b 1
)

REM Create build directory if it doesn't exist
if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

REM Configure the project
echo Configuring project...
cmake -B %BUILD_DIR% -G %CMAKE_GENERATOR% -A x64

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b %ERRORLEVEL%
)

REM Build the project
echo Building project...
cmake --build %BUILD_DIR% --config Release

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b %ERRORLEVEL%
)

echo Build completed successfully!
echo Executables can be found in: %BUILD_DIR%\Release\
