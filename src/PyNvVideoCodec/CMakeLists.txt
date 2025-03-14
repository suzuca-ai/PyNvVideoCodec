# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.21)

project(PyNvVideoCodec VERSION 1.0)

set(PYNVVIDEOCODEC_VERSION_MAJOR 1)
set(PYNVVIDEOCODEC_VERSION_MINOR 0)

if(POLICY CMP0135)
  #  https://cmake.org/cmake/help/latest/policy/CMP0135.html
  # From docs:
  # CMake 3.24 and above prefers to set the timestamps of all extracted contents to the time of the extraction.
  # This ensures that anything that depends on the extracted contents will be rebuilt whenever the URL changes.
  cmake_policy(SET CMP0135 NEW)
endif()

configure_file("inc/Version.hpp.in" "pynvcode_version.h")

find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter Development)


option(FETCHCONTENT_QUIET OFF)
include(FetchContent)
fetchcontent_declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.10.0.tar.gz
    URL_HASH
        SHA256=eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec
)
fetchcontent_makeavailable(pybind11)

message(STATUS "*****=\"${FFMPEG_DIR}/")
if(DEMUX_ONLY)

    set(PY_SOURCES
        src/PyNvDemuxer.cpp
        src/NvDemuxer.cpp
    )
    set(PY_HDRS
        inc
        ../VideoCodecSDKUtils/helper_classes/Utils
        
    )

else()

  set(PY_SOURCES
        src/DLPackUtils.cpp
        src/ExternalBuffer.cpp
        src/PyNvVideoCodec.cpp
        src/PyNvDemuxer.cpp
        src/PyNvEncoder.cpp
        src/NvDemuxer.cpp
        src/PyCAIMemoryView.cpp
        src/PyNvDecoder.cpp
        src/NvEncoderClInterface.cpp
        ../VideoCodecSDKUtils/helper_classes/NvCodec/NvEncoder/NvEncoderCuda.cpp
    )
    set(PY_HDRS
        inc
        ../VideoCodecSDKUtils/helper_classes/NvCodec/NvEncoder
        ${FFMPEG_DIR}/include/
    )

endif()

pybind11_add_module(_PyNvVideoCodec MODULE 
    ${PY_SOURCES}
)

set_property(TARGET _PyNvVideoCodec PROPERTY CXX_STANDARD 17)
target_include_directories(_PyNvVideoCodec 
    PUBLIC 
    ${PY_HDRS}
)

include(FetchContent)
FetchContent_Populate(
  dlpack
  URL        https://github.com/dmlc/dlpack/archive/refs/tags/v0.8.zip
  SOURCE_DIR dlpack
)
message(STATUS " downloading dlpack library " "${dlpack_SOURCE_DIR}")
target_include_directories(_PyNvVideoCodec PRIVATE "${dlpack_SOURCE_DIR}/include")


if(DEMUX_ONLY)
    target_compile_definitions(_PyNvVideoCodec PUBLIC -DDEMUX_ONLY=1)
else()
    if(USE_NVTX)
    message(STATUS "downloading nvtx library")
    option(FETCHCONTENT_QUIET OFF)
    include(FetchContent)
    fetchcontent_declare(
        nvtx_cpp
        URL https://github.com/NVIDIA/NVTX/archive/v3.1.0.tar.gz # Oct 5 2022
        URL_HASH
        SHA256=dc4e4a227d04d3da46ad920dfee5f7599ac8d6b2ee1809c9067110fb1cc71ced
        SOURCE_SUBDIR c
        DOWNLOAD_EXTRACT_TIMESTAMP ON)
    fetchcontent_makeavailable(nvtx_cpp)
    target_link_libraries(_PyNvVideoCodec PRIVATE nvtx3-cpp)
    target_compile_definitions(_PyNvVideoCodec PRIVATE -DUSE_NVTX=1)
    endif()
    target_link_libraries(_PyNvVideoCodec PUBLIC VideoCodecSDKUtils)
endif()

if(DEMUX_ONLY)
    include(../cmake/ffmpeg.cmake)
    if(NV_FFMPEG_HAS_BSF)
        target_compile_definitions(_PyNvVideoCodec PUBLIC -DHAS_BSF=1)
    endif()
    target_include_directories(_PyNvVideoCodec PUBLIC ${NV_FFMPEG_INCLUDE_DIR})
    target_link_libraries(_PyNvVideoCodec PUBLIC ${NV_FFMPEG_LIBRARIES})
endif()

set_target_properties(_PyNvVideoCodec PROPERTIES INSTALL_RPATH "$ORIGIN")

include(GNUInstallDirs)
# Install runtime dependencies (i.e. FFMPEG, nppi DLLS) on Windows but not Linux



if(WIN32)
  message(STATUS "FFMPEG_DIR/bin/=${FFMPEG_DIR}/bin/")
  install(TARGETS _PyNvVideoCodec
      RUNTIME_DEPENDENCIES DIRECTORIES "${FFMPEG_DIR}/lib/x64/"
               PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "python" "nvcuda"
               POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
      RUNTIME DESTINATION PyNvVideoCodec
      LIBRARY DESTINATION PyNvVideoCodec
  )
else()
  install(TARGETS _PyNvVideoCodec
      RUNTIME DESTINATION PyNvVideoCodec
      LIBRARY DESTINATION PyNvVideoCodec
  )
  
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    install(
	DIRECTORY 
	"${FFMPEG_DIR}/lib/x86_64/" 
	DESTINATION PyNvVideoCodec)
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    install(
	DIRECTORY 
	"${FFMPEG_DIR}/lib/aarch64/" 
	DESTINATION PyNvVideoCodec)

endif()


endif()
