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

#include "PyNvEncoder.hpp"
#include "NvEncoderClInterface.hpp"
#include "PyCAIMemoryView.hpp"

#include "cuda.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>

using namespace std;
using namespace chrono;

namespace py = pybind11;

PyNvEncoder::PyNvEncoder( PyNvEncoder&& pyenvc)
    :m_encoder(std::move(pyenvc.m_encoder)), m_CUcontext(pyenvc.m_CUcontext), m_width(pyenvc.m_width), m_height(pyenvc.m_height), m_eBufferFormat(pyenvc.m_eBufferFormat),
    pCUStream(std::move(pyenvc.pCUStream)), m_vecFrameObj(pyenvc.m_vecFrameObj), m_mapPtr(pyenvc.m_mapPtr)
{
    
}

PyNvEncoder::PyNvEncoder(PyNvEncoder& pyenvc)
    :m_encoder(std::move(pyenvc.m_encoder)), m_CUcontext(pyenvc.m_CUcontext), m_width(pyenvc.m_width), m_height(pyenvc.m_height), m_eBufferFormat(pyenvc.m_eBufferFormat),
    pCUStream(std::move(pyenvc.pCUStream)), m_vecFrameObj(pyenvc.m_vecFrameObj), m_mapPtr(pyenvc.m_mapPtr)
{

}

PyNvEncoder::PyNvEncoder(
        int _width,
        int _height,
        std::string _format,
        size_t  _cudacontext,
        size_t _cudastream,
        bool bUseCPUInutBuffer,
        std::map<std::string, std::string> kwargs)
{
    NV_ENC_BUFFER_FORMAT eBufferFormat;
    int iGPU = 0;
    CUcontext cudacontext =(CUcontext) _cudacontext;
    CUstream cudastream = (CUstream)_cudastream;

    NV_ENC_INITIALIZE_PARAMS params = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
    params.encodeConfig = &encodeConfig;

    if(_format == "NV12")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    }
    else if(_format == "ARGB")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
    }
    else if(_format == "ABGR")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
    }
    else if(_format == "YUV444")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_YUV444;
    }
    else if(_format == "YUV444_10BIT" || _format == "YUV444_16BIT")
    {
        _format = "YUV444_10BIT";
        eBufferFormat = NV_ENC_BUFFER_FORMAT_YUV444_10BIT;
    }
    else if(_format == "P010")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
    } 
    else if(_format == "ARGB10")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_ARGB10;
    }
    else if(_format == "ABGR10")
    {
         eBufferFormat = NV_ENC_BUFFER_FORMAT_ABGR10;
    }
    else if(_format == "YUV420")
    {
        eBufferFormat = NV_ENC_BUFFER_FORMAT_YV12;
    }
    else
    {
        throw std::invalid_argument("Error. Unsupported format. Supported formats: NV12, ARGB, ABGR, P010, YUV444, YUV444_10BIT");
    }
    params.bufferFormat = eBufferFormat;

    cuInit(0);
    if(cudacontext)
    {
        uint32_t version = 0;
        CUDA_DRVAPI_CALL(cuCtxGetApiVersion(cudacontext,&version));
	std::cout << "context created=" << cudacontext << std::endl;
    }
    else
    {
        CUDA_DRVAPI_CALL(cuCtxGetCurrent(&cudacontext));
	std::cout << "context fetched=" << cudacontext << std::endl;
        if(!cudacontext)
        {
            CUdevice cuDevice = 0;
            cuDeviceGet(&cuDevice, iGPU);
            CUDA_DRVAPI_CALL(cuCtxCreate(&cudacontext, 0, cuDevice));
            CUDA_DRVAPI_CALL(cuCtxPopCurrent(&cudacontext));
            m_bDestroyContext = true;
        }

    }

    if(!cudacontext)
    {
        throw std::runtime_error("Failed to create a cuda context. Create a cudacontext and pass it as named argument 'cudacontext = app_ctx'");
    }

    if(cudastream)
    {
        CUcontext streamCtx;
        CUDA_DRVAPI_CALL(cuStreamGetCtx(cudastream, &streamCtx));
        if(streamCtx != cudacontext)
        {
            throw std::invalid_argument("cudastream input argument does not correspond to cudacontext argument");
        }
     
    }
    else
   {
      	
	CUDA_DRVAPI_CALL(cuCtxPushCurrent(cudacontext));
    	CUDA_DRVAPI_CALL(cuStreamCreate(&cudastream, CU_STREAM_NON_BLOCKING););
    	CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
	std::cout << "stream created=" << cudastream<< std::endl;
   }


    m_encoder = std::make_unique<NvEncoderCuda>(cudacontext, cudastream,_width, _height, eBufferFormat);
    

    std::map<std::string,std::string> options = kwargs;
    options.insert({"fmt", _format});
    options.insert({"s", std::to_string(_width) + "x" + std::to_string(_height)});
    NvEncoderClInterface cliInterface(options);
    cliInterface.SetupInitParams(params, false, m_encoder->GetApi(), m_encoder->GetEncoder(), false);

    m_encoder->CreateEncoder(&params);
    pCUStream.reset(new NvCUStream(cudacontext, cudastream, m_encoder));
    InitEncodeReconfigureParams(params);
    m_CUcontext = cudacontext;
    m_CUstream = cudastream;
    m_width = _width;
    m_height = _height;
    m_eBufferFormat = eBufferFormat;
    m_bUseCPUInutBuffer = bUseCPUInutBuffer;
    m_vecFrameObj.clear();
    m_mapPtr.clear();
}

void PyNvEncoder::InitEncodeReconfigureParams(const NV_ENC_INITIALIZE_PARAMS params)
{
    NV_ENC_RC_PARAMS& reconfigRCParams = params.encodeConfig->rcParams;

    m_EncReconfigureParams.rateControlMode = reconfigRCParams.rateControlMode;
    m_EncReconfigureParams.multiPass = reconfigRCParams.multiPass;
    m_EncReconfigureParams.averageBitrate = reconfigRCParams.averageBitRate;
    m_EncReconfigureParams.vbvBufferSize = reconfigRCParams.vbvBufferSize;
    m_EncReconfigureParams.maxBitRate = reconfigRCParams.maxBitRate;
    m_EncReconfigureParams.vbvInitialDelay = reconfigRCParams.vbvInitialDelay;
    m_EncReconfigureParams.frameRateNum = params.frameRateNum;
    m_EncReconfigureParams.frameRateDen = params.frameRateDen;
}

structEncodeReconfigureParams PyNvEncoder::GetEncodeReconfigureParams()
{
    structEncodeReconfigureParams reconfigureParams;
    reconfigureParams.rateControlMode = m_EncReconfigureParams.rateControlMode;
    reconfigureParams.multiPass = m_EncReconfigureParams.multiPass;
    reconfigureParams.averageBitrate = m_EncReconfigureParams.averageBitrate;
    reconfigureParams.vbvBufferSize = m_EncReconfigureParams.vbvBufferSize;
    reconfigureParams.maxBitRate = m_EncReconfigureParams.maxBitRate;
    reconfigureParams.vbvInitialDelay = m_EncReconfigureParams.vbvInitialDelay;
    reconfigureParams.frameRateNum = m_EncReconfigureParams.frameRateNum;
    reconfigureParams.frameRateDen = m_EncReconfigureParams.frameRateDen;
    return reconfigureParams;
}

NV_ENC_REGISTERED_PTR PyNvEncoder::RegisterInputFrame(const py::object obj, const CAIMemoryView frame)
{
    //todo: validate frame size, format
    auto found = m_mapPtr.find(frame.data);
    NV_ENC_REGISTERED_PTR regPtr = nullptr;
    if(found == m_mapPtr.end())
    {
        void*  data =(void*) frame.data;
        regPtr = m_encoder->RegisterResource((void*)data, NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, frame.shape[1], frame.shape[0], frame.stride[0], NV_ENC_BUFFER_FORMAT_NV12, NV_ENC_INPUT_IMAGE);//todo : stride[-0] and and stride[1] need to beequal
        m_mapPtr.insert({(CUdeviceptr)data, regPtr});
        m_vecFrameObj.push_back(obj);
        return regPtr;
    }

    return found->second;
}

const NvEncInputFrame* PyNvEncoder::GetEncoderInputFromCPUBuffer(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> framedata)
{
    auto encoderInputFrame = m_encoder->GetNextInputFrame();
    void* srcPtr = (void*)framedata.data(0);
    uint32_t srcStride = 0;
    uint32_t srcChromaOffsets[2];

    switch (m_eBufferFormat)
    {
    case NV_ENC_BUFFER_FORMAT_NV12:
    {   
        srcChromaOffsets[0] = m_width * m_height;
        break;
    }
    case NV_ENC_BUFFER_FORMAT_YUV444:
    {
        srcChromaOffsets[0] = (m_width * m_height);
        srcChromaOffsets[1] = 2 * (m_width * m_height);
        break;
    }
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    {
        srcChromaOffsets[0] = 2* (m_width * m_height);
        srcChromaOffsets[1] = 4 * (m_width * m_height);
        break;
    }
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    {
        srcChromaOffsets[0] = m_width * m_height;
        break;
    }
    case NV_ENC_BUFFER_FORMAT_YV12:
    {
        srcChromaOffsets[0] = (m_width * m_height);
        break;
    }
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    {
        throw std::runtime_error("ARGB10 format not supported in current release. Use YUV444_16BIT or P010");
        break;
    }
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ARGB:
    {
        srcChromaOffsets[0] = 0;
        break;
    }
    default:
        throw std::runtime_error("Format not supported");
    }

    NvEncoderCuda::CopyToDeviceFrame(m_CUcontext,
        (void*)srcPtr,
        srcStride,
        (CUdeviceptr)encoderInputFrame->inputPtr,
        (int)encoderInputFrame->pitch,
        m_encoder->GetEncodeWidth(),
        m_encoder->GetEncodeHeight(),
        CU_MEMORYTYPE_HOST,
        encoderInputFrame->bufferFormat,
        encoderInputFrame->chromaOffsets,
        encoderInputFrame->numChromaPlanes,
        false,
        nullptr,
        srcChromaOffsets
    );
    return encoderInputFrame;
}

const NvEncInputFrame* PyNvEncoder::GetEncoderInput(py::object frame)
{
    auto encoderInputFrame = m_encoder->GetNextInputFrame();
    void * srcPtr = nullptr;
    uint32_t srcStride = 0;
    uint32_t srcChromaOffsets[2];

    if(m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_NV12)
    {
        //YUV420_10BIT is actually P010 format
        if (frame.attr("__dlpack__"))
        {
            if (frame.attr("__dlpack_device__"))
            {
                py::tuple dlpackDevice = frame.attr("__dlpack_device__")().cast<py::tuple>();
                auto      devType = static_cast<DLDeviceType>(dlpackDevice[0].cast<int>());
                if (!IsCudaAccessible(devType))
                {
                    throw std::runtime_error("Only CUDA-accessible memory buffers can be wrapped");
                }
            }
            py::capsule cap = frame.attr("__dlpack__")(1).cast<py::capsule>();
            if (auto* tensor = static_cast<DLManagedTensor*>(cap.get_pointer()))
            {
                srcPtr = tensor->dl_tensor.data;//we got the luma pointer
                py::tuple shape(tensor->dl_tensor.ndim);//assuming its a CHW tensor, so tensor height should be 1.5 times actual height
                int64_t tensorWidth = tensor->dl_tensor.shape[1];
                int64_t tensorHeight = tensor->dl_tensor.shape[0];
                if (tensorHeight != (m_height * 1.5))
                {
                    std::string error = "Tensor height :";
                    error.append(std::to_string(tensorHeight));
                    error.append(" must be 1.5 times the actual height :");
                    error.append(std::to_string(m_height));
                    error.append(" passed to encoder.");
                    throw std::runtime_error(error);
                }
                srcStride = tensor->dl_tensor.strides[0];
                srcChromaOffsets[0] = m_width * m_height;
            }
        }
        else
        {
            CAIMemoryView yPlane = coerceToCudaArrayView(frame.attr("__getitem__")(0), m_eBufferFormat, m_width, m_height, 0);
            CAIMemoryView uvPlane = coerceToCudaArrayView(frame.attr("__getitem__")(1), m_eBufferFormat, m_width, m_height, 1);

            if (yPlane.stride[0] != uvPlane.stride[0])
            {
                throw std::invalid_argument("unsupporte, argument : strides of y and uv plane  are different");
            }
            srcPtr = (void*)yPlane.data;
            srcStride = yPlane.stride[0];
            if (uvPlane.data <= yPlane.data)
            {
                throw std::invalid_argument("Unsupported surface allocation. u plane must follow yplane. ");
            }
            srcChromaOffsets[0] = static_cast<uint32_t>(uvPlane.data - yPlane.data);
        }
    }
    else if(m_eBufferFormat == NV_ENC_BUFFER_FORMAT_ARGB || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_ABGR 
            || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_ARGB10 || m_eBufferFormat == NV_ENC_BUFFER_FORMAT_ABGR10)
    {
        CAIMemoryView argb = coerceToCudaArrayView(frame, m_eBufferFormat, m_width, m_height);
        srcPtr =(void*) argb.data;
        srcStride = argb.stride[0];
        srcChromaOffsets[0] = 0;
    }
    else if(m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 || 
            m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
    {
        CAIMemoryView yPlane = coerceToCudaArrayView(frame.attr("__getitem__")(0), m_eBufferFormat, m_width, m_height, 0);
        CAIMemoryView uPlane = coerceToCudaArrayView(frame.attr("__getitem__")(1), m_eBufferFormat, m_width, m_height, 1);
        CAIMemoryView vPlane = coerceToCudaArrayView(frame.attr("__getitem__")(2), m_eBufferFormat, m_width, m_height, 2);
        if(uPlane.stride[0]!= vPlane.stride[0])
        {
            throw std::invalid_argument("unsupported argument : strides of  u, v must match");
        }
        srcPtr = (void*) yPlane.data;
        srcStride = yPlane.stride[0];
        if(uPlane.data <= yPlane.data || vPlane.data <= uPlane.data)
        {
            throw std::invalid_argument("Incorrect surface allocation. u and v plane must follow yplane.");
        }
        srcChromaOffsets[0] = uPlane.data-yPlane.data;
        srcChromaOffsets[1] =  vPlane.data-yPlane.data;
    }
    else
    {
        throw std::invalid_argument("unsupported format.");
    }
    NvEncoderCuda::CopyToDeviceFrame(m_CUcontext, 
        (void*) srcPtr,
        srcStride,
        (CUdeviceptr) encoderInputFrame->inputPtr,
        (int) encoderInputFrame->pitch,
        m_encoder->GetEncodeWidth(),
        m_encoder->GetEncodeHeight(),
        CU_MEMORYTYPE_DEVICE,
        encoderInputFrame->bufferFormat,
        encoderInputFrame->chromaOffsets,
        encoderInputFrame->numChromaPlanes,
        false,
        m_CUstream,
        srcChromaOffsets
        );
    return encoderInputFrame;
}

void PyNvEncoder::ConvertFrameNumToTimestamp(std::vector<std::pair<uint64_t, std::vector<uint8_t>>> &vPacket)
{
    for(auto& packet : vPacket)
    {
        auto found = m_mapFrameNumToTimestamp.find(packet.first);
        if(found == m_mapFrameNumToTimestamp.end()) {
            throw std::runtime_error("[BUG] frame number not found in map");
        }
        packet.first = found->second;
        m_mapFrameNumToTimestamp.erase(found);
    }
}

std::vector<std::pair<uint64_t, std::vector<uint8_t>>> PyNvEncoder::Encode(py::object _frame, int64_t timestamp_ns)
{
    py::object frame = _frame;

    if(hasattr(frame, "cuda"))
    {
        frame = frame.attr("cuda")();
        GetEncoderInput(frame);
    }
    else
    {
        if (!m_bUseCPUInutBuffer)
        {
            throw std::runtime_error("incorrect usage of CPU inut buffer");
        }
        GetEncoderInputFromCPUBuffer(frame);
    }

    NV_ENC_PIC_PARAMS picParam = { 0 };
    picParam.inputTimeStamp = m_frameNum++;

    // If timestamp_ns is -1, use current time in nanoseconds
    if (timestamp_ns == -1) {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    }
    m_mapFrameNumToTimestamp[picParam.inputTimeStamp] = timestamp_ns;

    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> vOutput;
    m_encoder->EncodeFrame(vOutput, &picParam);
    ConvertFrameNumToTimestamp(vOutput);
    return vOutput;
}

std::vector<std::pair<uint64_t, std::vector<uint8_t>>> PyNvEncoder::Encode()
{
    //flush the encoder
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> vOutput;
    m_encoder->EndEncode(vOutput);
    ConvertFrameNumToTimestamp(vOutput);
    return vOutput;
}

void PyNvEncoder::UnregisterInputFrame(const CAIMemoryView frame)
{

}


PyNvEncoder::~PyNvEncoder()
{
    for(auto& item : m_mapPtr)
    {
        m_encoder->UnregisterInputResource(item.second);
    }

    if(m_vecFrameObj.size() > 16)
    {
        //throw std::runtime_error("calling everytime with new alloc is suboptimal. Use small set of alloc");
        //todo: enable this error once we implement optimized copy support
    }

    m_width = 0;
    m_height = 0;

    if(m_bDestroyContext)
    {
        m_encoder.reset();
        pCUStream.reset();

        CUDA_DRVAPI_CALL(cuCtxDestroy(m_CUcontext));
        m_bDestroyContext = false;
    }

    m_CUcontext = nullptr;
}

bool PyNvEncoder::Reconfigure(structEncodeReconfigureParams rcParamsToChange)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;
    m_encoder->GetInitializeParams(&initializeParams);

    NV_ENC_RC_PARAMS& reconfigRCParams = initializeParams.encodeConfig->rcParams;

    reconfigRCParams.rateControlMode = rcParamsToChange.rateControlMode;
    reconfigRCParams.multiPass = rcParamsToChange.multiPass;
    reconfigRCParams.averageBitRate = rcParamsToChange.averageBitrate;
    reconfigRCParams.vbvBufferSize = rcParamsToChange.vbvBufferSize;
    reconfigRCParams.maxBitRate = rcParamsToChange.maxBitRate;
    reconfigRCParams.vbvInitialDelay = rcParamsToChange.vbvInitialDelay;
    initializeParams.frameRateDen = rcParamsToChange.frameRateDen;
    initializeParams.frameRateNum = rcParamsToChange.frameRateNum;

    NV_ENC_RECONFIGURE_PARAMS reconfigureParams = { NV_ENC_RECONFIGURE_PARAMS_VER };
    memcpy(&reconfigureParams.reInitEncodeParams, &initializeParams, sizeof(initializeParams));
    
    NV_ENC_CONFIG reInitCodecConfig = { NV_ENC_CONFIG_VER };
    memcpy(&reInitCodecConfig, initializeParams.encodeConfig, sizeof(reInitCodecConfig));
    
    reconfigureParams.reInitEncodeParams.encodeConfig = &reInitCodecConfig;

    InitEncodeReconfigureParams(initializeParams);

    reconfigureParams.reInitEncodeParams.tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY;

    return m_encoder->Reconfigure(const_cast<NV_ENC_RECONFIGURE_PARAMS*>(&reconfigureParams));

}

void Init_PyNvEncoder(py::module& m)
{
    //Rate control modes - NV_ENC_PARAMS_RC_MODE 
    py::enum_<NV_ENC_PARAMS_RC_MODE>(m, "NV_ENC_PARAMS_RC_MODE", py::module_local())
        .ENUM_VALUE(NV_ENC_PARAMS_RC, CONSTQP)  /* 0x0 */
        .ENUM_VALUE(NV_ENC_PARAMS_RC, VBR)      /* 0x1 */
        .ENUM_VALUE(NV_ENC_PARAMS_RC, CBR);     /* 0x2 */

    //Multi Pass encoding
    py::enum_<NV_ENC_MULTI_PASS>(m, "NV_ENC_MULTI_PASS", py::module_local())
        .ENUM_VALUE(NV_ENC_MULTI_PASS, DISABLED)          /* 0x0 */
        .ENUM_VALUE(NV_ENC_TWO_PASS, QUARTER_RESOLUTION)  /* 0x1 */
        .ENUM_VALUE(NV_ENC_TWO_PASS, FULL_RESOLUTION);    /* 0x2 */

    py::class_<structEncodeReconfigureParams, std::shared_ptr<structEncodeReconfigureParams>>(m, "structEncodeReconfigureParams")
        .def(py::init<>())
        .def_readwrite("rateControlMode", &structEncodeReconfigureParams::rateControlMode)
        .def_readwrite("multiPass", &structEncodeReconfigureParams::multiPass)
        .def_readwrite("averageBitrate", &structEncodeReconfigureParams::averageBitrate)
        .def_readwrite("vbvBufferSize", &structEncodeReconfigureParams::vbvBufferSize)
        .def_readwrite("maxBitRate", &structEncodeReconfigureParams::maxBitRate)
        .def_readwrite("vbvInitialDelay", &structEncodeReconfigureParams::vbvInitialDelay)
        .def_readwrite("frameRateDen", &structEncodeReconfigureParams::frameRateDen)
        .def_readwrite("frameRateNum", &structEncodeReconfigureParams::frameRateNum)
        .def("__repr__",
            [](std::shared_ptr<structEncodeReconfigureParams>& self)
            {
                std::stringstream ss;
                ss << "Reconfig Params [";
                ss << "rateControlMode=" << self->rateControlMode;
                ss << ", multiPass=" << self->multiPass;
                ss << ", averageBitrate=" << self->averageBitrate;
                ss << ", vbvBufferSize=" << self->vbvBufferSize;
                ss << ", maxBitRate=" << self->maxBitRate;
                ss << ", vbvInitialDelay=" << self->vbvInitialDelay;
                ss << ", frameRateDen=" << self->frameRateDen;
                ss << ", frameRateNum=" << self->frameRateNum;
                ss << "]";
                return ss.str();
            })
        ;

    py::class_<PyNvEncoder, shared_ptr<PyNvEncoder>>(m, "PyNvEncoder", py::module_local())
        .def(py::init<int, int, std::string,  size_t , size_t,  bool ,std::map<std::string,std::string>>(),
            R"pbdoc(
                Constructor method. Initialize encoder session with set of particular paramters
                :param width, height, format, cpuinputbuffer,other-optional-params,  
            )pbdoc")
        .def(
             "Encode",
             [](std::shared_ptr<PyNvEncoder>& self, const py::object frame, int64_t timestamp_ns = -1)
             {
                return self->Encode(frame, timestamp_ns);
             }, R"pbdoc(
                 Encode frame. Returns encoded bitstream in CPU memory
                 :param frame: NVCV Image object or any object that implements __cuda_array_interface
                 :param timestamp_ns: Optional timestamp in nanoseconds. If not provided or -1, current time will be used.
             )pbdoc")
        .def(
             "EndEncode",
             [](std::shared_ptr<PyNvEncoder>& self)
             {
                return self->Encode();
             }, R"pbdoc(
                 Flush encoder to retreive bitstreams in the queue. Returns encoded bitstream in CPU memory
                 :param empty
             )pbdoc")
          .def(
               "CopyToDeviceMemory",
                     [](std::shared_ptr<PyNvEncoder>& self, const std::string& filePath)
                     {
                        uint8_t* pBuf = NULL;
                        uint64_t nBufSize = 0;
                        CUdeviceptr dpBuf = NULL;
                        BufferedFileReader bufferedFileReader(filePath.c_str(), true);
                        if (!bufferedFileReader.GetBuffer(&pBuf, &nBufSize)) {
                            std::cout << "Failed to read file " << filePath.c_str() << std::endl;
                            return dpBuf;
                        }

                        std::vector<CUdeviceptr> vdpBuf;
                        
                        ck(cuMemAlloc(&dpBuf, (size_t)nBufSize));
                        vdpBuf.push_back(dpBuf);
                        ck(cuMemcpyHtoD(dpBuf, pBuf, (size_t)nBufSize));
                        return dpBuf;

                     }, R"pbdoc(
                 Copies entire raw file from host memory to device memory
                 :param empty
             )pbdoc")

        .def("GetEncodeReconfigureParams", &PyNvEncoder::GetEncodeReconfigureParams,
              R"pbdoc(Get the values of reconfigure params, value to get )pbdoc")
       
        .def("Reconfigure", &PyNvEncoder::Reconfigure,
            R"pbdoc( Encode API called with new params :reconfigure params struct)pbdoc")
             ;
}
