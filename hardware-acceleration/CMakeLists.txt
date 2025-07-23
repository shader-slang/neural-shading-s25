cmake_minimum_required(VERSION 3.20)
project(neural-shading-hardware-acceleration)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include necessary CMake modules
include(FetchContent)
include(ExternalProject)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Function to download and extract a release from GitHub
function(download_github_release REPO_URL TAG_NAME TARGET_DIR)
    set(DOWNLOAD_URL "${REPO_URL}/releases/download/${TAG_NAME}")
    set(ARCHIVE_NAME "${TAG_NAME}.zip")
    set(DOWNLOAD_PATH "${CMAKE_BINARY_DIR}/downloads/${ARCHIVE_NAME}")
    
    # Create downloads directory
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/downloads")
    
    # Download the release if it doesn't exist
    if(NOT EXISTS ${DOWNLOAD_PATH})
        message(STATUS "Downloading ${REPO_URL} release ${TAG_NAME}...")
        file(DOWNLOAD 
            "${DOWNLOAD_URL}/${ARCHIVE_NAME}" 
            ${DOWNLOAD_PATH}
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
        )
        
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_ERROR)
        if(DOWNLOAD_ERROR)
            # Try alternative download URLs
            file(DOWNLOAD 
                "${REPO_URL}/archive/refs/tags/${TAG_NAME}.zip" 
                ${DOWNLOAD_PATH}
                SHOW_PROGRESS
                STATUS DOWNLOAD_STATUS_ALT
            )
            list(GET DOWNLOAD_STATUS_ALT 0 DOWNLOAD_ERROR_ALT)
            if(DOWNLOAD_ERROR_ALT)
                message(FATAL_ERROR "Failed to download ${REPO_URL} release ${TAG_NAME}")
            endif()
        endif()
    endif()
    
    # Extract the archive if target directory doesn't exist
    if(NOT EXISTS ${TARGET_DIR})
        message(STATUS "Extracting ${ARCHIVE_NAME} to ${TARGET_DIR}...")
        file(ARCHIVE_EXTRACT 
            INPUT ${DOWNLOAD_PATH}
            DESTINATION "${CMAKE_BINARY_DIR}/external"
        )
        # Find the extracted directory and rename it to target name
        file(GLOB EXTRACTED_DIRS "${CMAKE_BINARY_DIR}/external/*${TAG_NAME}*")
        if(EXTRACTED_DIRS)
            list(GET EXTRACTED_DIRS 0 EXTRACTED_DIR)
            file(RENAME ${EXTRACTED_DIR} ${TARGET_DIR})
        endif()
    endif()
endfunction()

# Download external dependencies
set(SLANG_VERSION "v2025.13")  # Adjust version as needed

set(SLANG_DIR "${CMAKE_BINARY_DIR}/external/slang")
set(SLANG_RHI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/slang-rhi")
set(LZ4_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/lz4")
set(MINIZ_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/miniz")
set(UNORDERED_DENSE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/unordered_dense")

# Download Slang release
download_github_release("https://github.com/shader-slang/slang" ${SLANG_VERSION} ${SLANG_DIR})

# Check if slang-rhi submodule exists and is initialized
if(NOT EXISTS "${SLANG_RHI_DIR}/CMakeLists.txt")
    message(FATAL_ERROR 
        "slang-rhi submodule not found. Please initialize it with:\n"
        "  git submodule add https://github.com/shader-slang/slang-rhi.git external/slang-rhi\n"
        "  git submodule update --init --recursive"
    )
endif()

# Check if lz4 submodule exists and is initialized
if(NOT EXISTS "${LZ4_DIR}/build/cmake/CMakeLists.txt")
    message(FATAL_ERROR 
        "lz4 submodule not found. Please initialize it with:\n"
        "  git submodule add https://github.com/lz4/lz4.git external/lz4\n"
        "  git submodule update --init --recursive"
    )
endif()

# Check if miniz submodule exists and is initialized
if(NOT EXISTS "${MINIZ_DIR}/CMakeLists.txt")
    message(FATAL_ERROR 
        "miniz submodule not found. Please initialize it with:\n"
        "  git submodule add https://github.com/richgel999/miniz.git external/miniz\n"
        "  git submodule update --init --recursive"
    )
endif()

# Check if unordered_dense submodule exists and is initialized
if(NOT EXISTS "${UNORDERED_DENSE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR 
        "unordered_dense submodule not found. Please initialize it with:\n"
        "  git submodule add https://github.com/martinus/unordered_dense.git external/unordered_dense\n"
        "  git submodule update --init --recursive"
    )
endif()

# Add slang-rhi as subdirectory (build from source)
add_subdirectory(${SLANG_RHI_DIR})

# Set slang-rhi to use C++17 while everything else uses C++20
set_target_properties(slang-rhi PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add unordered_dense as subdirectory (from submodule)
add_subdirectory(${UNORDERED_DENSE_DIR})

# Add miniz as subdirectory (from submodule)
# Configure miniz to not build examples/fuzzers when included as subdirectory
set(BUILD_EXAMPLES OFF CACHE BOOL "Build miniz examples" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "Build miniz fuzzers" FORCE)
add_subdirectory(${MINIZ_DIR})

# Add LZ4 as subdirectory (build from source)
# Configure LZ4 to build only the static library
set(LZ4_BUILD_CLI OFF CACHE BOOL "Build LZ4 command line tools" FORCE)
set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "Build legacy lz4c" FORCE)
add_subdirectory(${LZ4_DIR}/build/cmake)

# Find Slang headers (from downloaded release)
find_path(SLANG_INCLUDE_DIR 
    NAMES slang.h 
    PATHS 
        ${SLANG_DIR}/include
        ${SLANG_DIR}/slang/include
        ${SLANG_DIR}/source/slang/include
    NO_DEFAULT_PATH
)

# Set slang-rhi include directory (from submodule)
set(SLANG_RHI_INCLUDE_DIR "${SLANG_RHI_DIR}/include")

# Try to find pre-built Slang library
find_library(SLANG_LIBRARY 
    NAMES slang libslang
    PATHS 
        ${SLANG_DIR}/lib
        ${SLANG_DIR}/bin
        ${SLANG_DIR}/build/lib
        ${SLANG_DIR}/build/bin
    NO_DEFAULT_PATH
)

# If libraries are not found, we'll need to build them or use header-only approach
if(NOT SLANG_INCLUDE_DIR)
    message(WARNING "Slang headers not found, using fallback paths")
    set(SLANG_INCLUDE_DIR ${SLANG_DIR})
endif()

# Verify slang-rhi include directory exists
if(NOT EXISTS ${SLANG_RHI_INCLUDE_DIR})
    message(FATAL_ERROR "Slang-RHI include directory not found: ${SLANG_RHI_INCLUDE_DIR}")
endif()

# Collect all source files for example-base static library
file(GLOB_RECURSE EXAMPLE_BASE_SOURCES
    example-base/*.cpp
    example-base/*.h
)

# Filter out platform-specific files we don't need
list(FILTER EXAMPLE_BASE_SOURCES EXCLUDE REGEX ".*/apple/.*")
list(FILTER EXAMPLE_BASE_SOURCES EXCLUDE REGEX ".*/linux/.*")
list(FILTER EXAMPLE_BASE_SOURCES EXCLUDE REGEX ".*/placeholder/.*")

# Filter out unordered_dense files (it's built separately as a submodule)
list(FILTER EXAMPLE_BASE_SOURCES EXCLUDE REGEX ".*/unordered_dense/.*")

# Filter out miniz files (it's built separately as a submodule)
list(FILTER EXAMPLE_BASE_SOURCES EXCLUDE REGEX ".*/miniz/.*")

# Debug: print remaining source files
message(STATUS "Example-base source files:")
foreach(SOURCE_FILE ${EXAMPLE_BASE_SOURCES})
    message(STATUS "  ${SOURCE_FILE}")
endforeach()

# Only include Windows platform files on Windows
if(NOT WIN32)
    list(FILTER EXAMPLE_BASE_SOURCES EXCLUDE REGEX ".*/windows/.*")
endif()

# Create example-base static library
add_library(example-base STATIC ${EXAMPLE_BASE_SOURCES})

target_include_directories(example-base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/example-base
    ${CMAKE_CURRENT_SOURCE_DIR}/example-base/core
    ${CMAKE_CURRENT_SOURCE_DIR}/example-base/platform
    ${SLANG_INCLUDE_DIR}
    ${SLANG_RHI_INCLUDE_DIR}
    ${SLANG_RHI_DIR}/include  # Additional fallback
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(example-base PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        UNICODE
        _UNICODE
    )
    target_link_libraries(example-base PUBLIC
        user32
        gdi32
        shell32
        d3d12
        dxgi
        dxguid
    )
endif()

# Link external libraries if found
if(SLANG_LIBRARY)
    target_link_libraries(example-base PUBLIC ${SLANG_LIBRARY})
endif()

# Link with slang-rhi built from submodule
target_link_libraries(example-base PUBLIC slang-rhi)

# Link with unordered_dense header-only library
target_link_libraries(example-base PUBLIC unordered_dense::unordered_dense)

# Link with miniz compression library
target_link_libraries(example-base PUBLIC miniz)

# Link with LZ4 compression library
target_link_libraries(example-base PUBLIC lz4_static)

# Set compiler-specific flags
if(MSVC)
    target_compile_options(example-base PRIVATE
        /W3
        /wd4996  # Disable deprecated function warnings
        /wd4267  # Disable size_t conversion warnings
        /wd4244  # Disable type conversion warnings
    )
else()
    target_compile_options(example-base PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-missing-field-initializers
    )
endif()

# Function to add MLP training executable
function(add_mlp_executable TARGET_NAME SOURCE_DIR)
    # Collect source files
    file(GLOB SOURCES ${SOURCE_DIR}/*.cpp)
    file(GLOB SLANG_FILES ${SOURCE_DIR}/*.slang)
    
    # Create executable
    add_executable(${TARGET_NAME} ${SOURCES})
    
    # Link with example-base library
    target_link_libraries(${TARGET_NAME} PRIVATE example-base)
    
    # Include directories
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/example-base
        ${SLANG_INCLUDE_DIR}
        ${SLANG_RHI_INCLUDE_DIR}
    )
    
    # Copy .slang files to output directory
    foreach(SLANG_FILE ${SLANG_FILES})
        get_filename_component(SLANG_FILENAME ${SLANG_FILE} NAME)
        configure_file(${SLANG_FILE} ${CMAKE_BINARY_DIR}/${SLANG_FILENAME} COPYONLY)
    endforeach()
    
    # Set target properties
    set_target_properties(${TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${TARGET_NAME}
        DEBUG_POSTFIX ""
    )
    
    # Platform-specific settings
    if(WIN32)
        # Build as console application, not windowed application
        set_target_properties(${TARGET_NAME} PROPERTIES
            WIN32_EXECUTABLE FALSE
        )
        target_compile_definitions(${TARGET_NAME} PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX
            _CRT_SECURE_NO_WARNINGS
            SLANG_EXAMPLE_NAME=${TARGET_NAME}
        )
    endif()
    
    # Set runtime path for shared libraries
    if(UNIX)
        set_target_properties(${TARGET_NAME} PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()
endfunction()

# Add the two MLP training executables
add_mlp_executable(mlp-training mlp-training)
add_mlp_executable(mlp-training-coopvec mlp-training-coopvec)

# Install targets
install(TARGETS mlp-training mlp-training-coopvec example-base
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install shader files
install(DIRECTORY mlp-training/ 
    DESTINATION bin
    FILES_MATCHING PATTERN "*.slang"
)

install(DIRECTORY mlp-training-coopvec/ 
    DESTINATION bin
    FILES_MATCHING PATTERN "*.slang"
)

# Print configuration info
message(STATUS "Build configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Slang include dir: ${SLANG_INCLUDE_DIR}")
message(STATUS "  Slang-RHI include dir: ${SLANG_RHI_INCLUDE_DIR}")
if(SLANG_LIBRARY)
    message(STATUS "  Slang library: ${SLANG_LIBRARY}")
endif()
message(STATUS "  Slang-RHI: Building from submodule")
message(STATUS "  unordered_dense: Building from submodule")
message(STATUS "  miniz: Building from submodule")
message(STATUS "  LZ4: Building from submodule")
