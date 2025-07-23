@echo off
REM Setup script for Neural Shading Hardware Acceleration
REM This script initializes the required Git submodules

echo Setting up project dependencies...

REM Check if we're in a git repository
git status >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: This is not a Git repository
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

REM Check if external directory exists
if not exist external (
    mkdir external
)

REM Add slang-rhi submodule if it doesn't exist
if not exist external\slang-rhi (
    echo Adding slang-rhi submodule...
    git submodule add https://github.com/shader-slang/slang-rhi.git external/slang-rhi
    if %ERRORLEVEL% neq 0 (
        echo Failed to add slang-rhi submodule
        pause
        exit /b %ERRORLEVEL%
    )
)

REM Add lz4 submodule if it doesn't exist
if not exist external\lz4 (
    echo Adding lz4 submodule...
    git submodule add https://github.com/lz4/lz4.git external/lz4
    if %ERRORLEVEL% neq 0 (
        echo Failed to add lz4 submodule
        pause
        exit /b %ERRORLEVEL%
    )
)

REM Add miniz submodule if it doesn't exist
if not exist external\miniz (
    echo Adding miniz submodule...
    git submodule add https://github.com/richgel999/miniz.git external/miniz
    if %ERRORLEVEL% neq 0 (
        echo Failed to add miniz submodule
        pause
        exit /b %ERRORLEVEL%
    )
)

REM Add unordered_dense submodule if it doesn't exist
if not exist external\unordered_dense (
    echo Adding unordered_dense submodule...
    git submodule add https://github.com/martinus/unordered_dense.git external/unordered_dense
    if %ERRORLEVEL% neq 0 (
        echo Failed to add unordered_dense submodule
        pause
        exit /b %ERRORLEVEL%
    )
)

REM Initialize and update all submodules
echo Initializing and updating submodules...
git submodule update --init --recursive
if %ERRORLEVEL% neq 0 (
    echo Failed to update submodules
    pause
    exit /b %ERRORLEVEL%
)

echo Setup completed successfully!
echo You can now run build.bat to build the project.
