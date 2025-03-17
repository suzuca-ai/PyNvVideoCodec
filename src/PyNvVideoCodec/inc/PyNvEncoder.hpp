/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <map>
#include <optional>

#include "NvEncoderCuda.h"
#include "PyCAIMemoryView.hpp"

namespace py = pybind11;

// This class allocates CUStream.
// It also sets the input and output CUDA stream in the driver, which will be used for pipelining
// pre and post processing CUDA tasks
class NvCUStream
{
public:
    NvCUStream(CUcontext cuDevice, CUstream cuStream, std::unique_ptr<NvEncoderCuda>& pEnc)
    {
        device = cuDevice;
        CUDA_DRVAPI_CALL(cuCtxPushCurrent(device));

        if (cuStream == 0)
        {
            ck(cuStreamCreate(&inputStream, CU_STREAM_DEFAULT));
        }
        else
        {
            inputStream = cuStream;
        }

        outputStream = inputStream;

        CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));

        // Set input and output CUDA streams in driver
        pEnc->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&inputStream, (NV_ENC_CUSTREAM_PTR)&outputStream);
    }

    ~NvCUStream()
    {
        ck(cuCtxPushCurrent(device));

        if (inputStream == outputStream)
        {
            if (inputStream != NULL)
                ck(cuStreamDestroy(inputStream));
        }
        else
        {
            if (inputStream != NULL)
                ck(cuStreamDestroy(inputStream));

            if (outputStream != NULL)
                ck(cuStreamDestroy(outputStream));
        }

        ck(cuCtxPopCurrent(NULL));
    }

    CUstream GetOutputCUStream() { return outputStream; };
    CUstream GetInputCUStream() { return inputStream; };

private:
    CUcontext device;
    CUstream inputStream = NULL, outputStream = NULL;
};


struct structEncodeReconfigureParams
{
    NV_ENC_PARAMS_RC_MODE  rateControlMode;
    NV_ENC_MULTI_PASS multiPass;
    uint32_t averageBitrate;
    uint32_t vbvBufferSize ;
    uint32_t maxBitRate;
    uint32_t vbvInitialDelay;
    uint32_t frameRateNum;
    uint32_t frameRateDen;
};

class PyNvEncoder {
private:
    CUcontext m_CUcontext = nullptr;
    CUstream m_CUstream = nullptr;
    bool m_bDestroyContext = false;
    std::map<CUdeviceptr, NV_ENC_REGISTERED_PTR> m_mapPtr;
    std::vector<py::object> m_vecFrameObj;
    size_t m_width;
    size_t m_height;
    uint64_t m_frameNum;
    std::unordered_map<uint64_t, uint64_t> m_mapFrameNumToTimestamp;
    NV_ENC_BUFFER_FORMAT m_eBufferFormat;
    bool m_bUseCPUInutBuffer;

    const NvEncInputFrame* GetEncoderInput(py::object _frame);
    const NvEncInputFrame* GetEncoderInputFromCPUBuffer(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> _frame);
    void ConvertFrameNumToTimestamp(std::vector<NvEncOutputBitstream> &vPacket);
    std::unique_ptr<NvCUStream> pCUStream;
    structEncodeReconfigureParams m_EncReconfigureParams;
protected:
    std::unique_ptr<NvEncoderCuda> m_encoder;

public:
    explicit PyNvEncoder(int width, int height,  std::string format,
            size_t cudastream, size_t cudacontext, bool bUseCPUInutBuffer,std::map<std::string, std::string> config);
    PyNvEncoder(PyNvEncoder&& pyenvc);
    PyNvEncoder(PyNvEncoder& pyenvc);
    NV_ENC_REGISTERED_PTR RegisterInputFrame(const py::object obj, const CAIMemoryView frame); 
    bool Reconfigure(structEncodeReconfigureParams reconfigureParams);
    std::vector<NvEncOutputBitstream> Encode(const py::object frame, std::optional<int64_t> timestamp_ns = std::nullopt);
    std::vector<NvEncOutputBitstream> Encode();
    void UnregisterInputFrame(const CAIMemoryView frame);
    void InitEncodeReconfigureParams(const NV_ENC_INITIALIZE_PARAMS params);
    structEncodeReconfigureParams GetEncodeReconfigureParams();

     ~PyNvEncoder();
};
