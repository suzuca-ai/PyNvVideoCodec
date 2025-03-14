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

#include "PyNvDecoder.hpp"
#include "ExternalBuffer.hpp"

using namespace std;
using time_point_ms
= std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>;

namespace py = pybind11;

PyNvDecoder::PyNvDecoder(
    int _gpuid,
    cudaVideoCodec _codec,
    size_t _context,
    size_t _stream,
    bool m_bUseDeviceFrame,
    bool _enableasyncallocations
) : m_bDestroyContext(false)
{
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (_gpuid < 0 || _gpuid >= nGpu) {
        std::ostringstream err;
        err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        throw std::invalid_argument(err.str());
    }

    if(_context)
    {
        uint32_t version = 0;
        cuContext =reinterpret_cast<CUcontext>( _context);
        ck(cuCtxGetApiVersion(cuContext, &version));
    }
    else
    {
        ck(cuCtxGetCurrent(&cuContext));
        if(!cuContext)
        {
            CUdevice cuDevice = 0;
            ck(cuDeviceGet(&cuDevice, 0));
            createCudaContext(&cuContext, _gpuid, 0);
            ck(cuCtxPopCurrent(&cuContext));
            m_bDestroyContext = true;
        }
        //ck(cuCtxPopCurrent(&cuContext));
    }

    if(!cuContext)
    {
        throw std::runtime_error("Failed to create a cuda context. Create a cudacontext and pass it as named argument 'cudacontext = app_ctx'");
    }

    if(_stream)
    {
        CUcontext streamCtx;
        cuStream = reinterpret_cast<CUstream>(_stream);
        cuStreamGetCtx(cuStream, &streamCtx);
        if(streamCtx != cuContext)
        {
            throw std::invalid_argument("cudastream input argument does not correspond to cudacontext argument");
        }

    }

    decoder.reset(new NvDecoder(cuStream, cuContext, m_bUseDeviceFrame, _codec,false,_enableasyncallocations, false));

}

PyNvDecoder::~PyNvDecoder()
{
    decoder.reset();
    if (m_bDestroyContext)
    {
        ck(cuCtxDestroy(cuContext));
    }
}

Pixel_Format PyNvDecoder::GetNativeFormat(const cudaVideoSurfaceFormat inputFormat)
{
    switch (inputFormat)
    {
    case cudaVideoSurfaceFormat_NV12: return Pixel_Format_NV12;
    case cudaVideoSurfaceFormat_P016: return Pixel_Format_P016;
    case cudaVideoSurfaceFormat_YUV444: return Pixel_Format_YUV444;
    case cudaVideoSurfaceFormat_YUV444_16Bit: return Pixel_Format_YUV444_16Bit;
    default:
        break;
    }
    return Pixel_Format_UNDEFINED;
}

uint8_t* PyNvDecoder::GetLockedFrame(int64_t* pTimestamp)
{
    NVTX_SCOPED_RANGE("py::GetLockedFrame")
    uint8_t* framePtr = (decoder->GetLockedFrame(pTimestamp));
    
    return framePtr;
}

void PyNvDecoder::UnlockFrame(uint8_t* framePtr)
{
    NVTX_SCOPED_RANGE("py::UnlockFrame")
    decoder->UnlockFrame(framePtr);
}

int PyNvDecoder::GetNumDecodedFrame(const PacketData packetData)
{
    NVTX_SCOPED_RANGE("py::GetNumDecodedFrame")
    int  numFrames = decoder->Decode((uint8_t*)packetData.bsl_data, packetData.bsl, 0);
    return numFrames;
}


std::vector<DecodedFrame> PyNvDecoder::Decode(const PacketData packetData)
{
    NVTX_SCOPED_RANGE("py::decode")
    std::vector<DecodedFrame> frames;
    auto vecTupFrame = decoder->Decode((uint8_t*)packetData.bsl_data, packetData.bsl);
    std::transform(vecTupFrame.begin(), vecTupFrame.end(), std::back_inserter(frames),
        [=](std::tuple<CUdeviceptr, int64_t> tup)
        {
            DecodedFrame frame;

            frame.format = GetNativeFormat(decoder->GetOutputFormat());
            auto width = size_t(decoder->GetWidth());
            auto height = size_t(decoder->GetHeight());
            auto data = std::get<0>(tup);
            auto timestamp = std::get<1>(tup);
            frame.timestamp = timestamp;
            switch (frame.format)
            {
                case Pixel_Format_NV12:
                {
                    frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u1",reinterpret_cast<size_t>( decoder->GetStream()) ,(data), false });
                    frame.views.push_back(CAIMemoryView{ {height / 2, width / 2, 2}, {width / 2 * 2, 2, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()),(data + width * height), false });//todo: data+width*height assumes both planes are contiguous. Actual NVENC allocation can have padding?
                    // Load DLPack Tensor
                    std::vector<size_t> shape{ (size_t)(height * 1.5), width};
                    std::vector<size_t> stride{ size_t(width), 1};
                    int returntype = frame.extBuf->LoadDLPack( shape, stride, "|u1", reinterpret_cast<size_t>(decoder->GetStream()), data, false );
                }
                break;
                case Pixel_Format_P016:
                {
                    frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
                    frame.views.push_back(CAIMemoryView{ {height / 2, width / 2, 2}, {width / 2 * 2, 2, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()),(data + 2*(width * height)), false });//todo: data+width*height assumes both planes are contiguous. Actual NVENC allocation can have padding?
                }
                break;
                case Pixel_Format_YUV444:
                {
                    frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
                    frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()),(data + width * height), false });//todo: data+width*height assumes both planes are contiguous. Actual NVENC allocation can have padding?
                    frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u1",reinterpret_cast<size_t>(decoder->GetStream()),(data + 2 * (width * height)), false });//todo: data+width*height assumes both planes are contiguous. Actual NVENC allocation can have padding?
                }
                case Pixel_Format_YUV444_16Bit:
                {
                    frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()) ,(data), false });
                    frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()),(data + 2 * (width * height)), false });//todo: data+width*height assumes both planes are contiguous. Actual NVENC allocation can have padding?
                    frame.views.push_back(CAIMemoryView{ {height, width, 1}, {width, 1, 1}, "|u2",reinterpret_cast<size_t>(decoder->GetStream()),(data + 4 * (width * height)), false });//todo: data+width*height assumes both planes are contiguous. Actual NVENC allocation can have padding?
                }
                
            }
            
            return frame;
        });
    return frames;
}

void Init_PyNvDecoder(py::module& m)
{
    py::enum_<cudaVideoCodec>(m, "cudaVideoCodec", py::module_local())
        .ENUM_VALUE(cudaVideoCodec, MPEG1)
        .ENUM_VALUE(cudaVideoCodec, MPEG2)    /**<  MPEG2             */
        .ENUM_VALUE(cudaVideoCodec, MPEG4)    /**<  MPEG4             */
        .ENUM_VALUE(cudaVideoCodec, VC1)      /**<  VC1               */
        .ENUM_VALUE(cudaVideoCodec, H264)     /**<  H264              */
        .ENUM_VALUE(cudaVideoCodec, JPEG)     /**<  JPEG              */
        .ENUM_VALUE(cudaVideoCodec, H264_SVC) /**<  H264-SVC          */
        .ENUM_VALUE(cudaVideoCodec, H264_MVC) /**<  H264-MVC          */
        .ENUM_VALUE(cudaVideoCodec, HEVC)     /**<  HEVC              */
        .ENUM_VALUE(cudaVideoCodec, VP8)      /**<  VP8               */
        .ENUM_VALUE(cudaVideoCodec, VP9)      /**<  VP9               */
        .ENUM_VALUE(cudaVideoCodec, AV1)      /**<  AV1               */
        //.ENUM_VALUE(cudaVideoCodec_, NumCodecs) /**<  Max codecs        */
        // Uncompressed YUV
        .ENUM_VALUE(cudaVideoCodec, YUV420)
        .ENUM_VALUE(cudaVideoCodec, YV12)
        .ENUM_VALUE(cudaVideoCodec, NV12)
        .ENUM_VALUE(cudaVideoCodec, YUYV)
        .ENUM_VALUE(cudaVideoCodec, UYVY);

    py::enum_<cudaVideoSurfaceFormat>(m, "cudaVideoSurfaceFormat", py::module_local())
        .ENUM_VALUE(cudaVideoSurfaceFormat, NV12)
        .ENUM_VALUE(cudaVideoSurfaceFormat, P016)
        .ENUM_VALUE(cudaVideoSurfaceFormat, YUV444)
        .ENUM_VALUE(cudaVideoSurfaceFormat, YUV444_16Bit);

    py::enum_<Pixel_Format>(m, "Pixel_Format", py::module_local())
        .ENUM_VALUE(Pixel_Format, NV12)
        .ENUM_VALUE(Pixel_Format, YUV444)
        .ENUM_VALUE(Pixel_Format, P016)
        .ENUM_VALUE(Pixel_Format, YUV444_16Bit);

    
    m.def(
        "CreateDecoder",
        [](
            int gpuid,
            cudaVideoCodec codec,
            size_t cudacontext,
            size_t cudastream,
            bool usedevicememory
            )
        {
            return std::make_shared<PyNvDecoder>(0, codec, cudacontext, cudastream, usedevicememory, false);
        },

        py::arg("gpuid") = 0,
            py::arg("codec") = cudaVideoCodec::cudaVideoCodec_H264,
            py::arg("cudacontext") = 0,
            py::arg("cudastream") = 0,
            py::arg("usedevicememory") = 0,
            R"pbdoc(
        Initialize decoder with set of particular
        parameters
        :param gpuid: GPU Id
        :param codec : Video Codec
        :param context : CUDA context
        :param stream : CUDA Stream
        :param use_device_memory : decoder output surface is in device memory if true else on host memory
    )pbdoc"
            )
        .def(
            "CreateDecoder",
            [](
                int gpuid,
                cudaVideoCodec codec,
                size_t cudacontext,
                size_t cudastream,
                bool usedevicememory,
                bool enableasyncallocations
                )
            {
                return std::make_shared<PyNvDecoder>(0, codec, cudacontext, cudastream, true, enableasyncallocations);
            },

            py::arg("gpuid") = 0,
                py::arg("codec") = cudaVideoCodec::cudaVideoCodec_H264,
                py::arg("cudacontext") = 0,
                py::arg("cudastream") = 0,
                py::arg("usedevicememory") = 1,
                py::arg("enableasyncallocations") = 1,

                R"pbdoc(
        Initialize decoder with set of particular
        parameters
        :param gpuid: GPU Id
        :param codec : Video Codec
        :param context : CUDA context
        :param stream : CUDA Stream
        :param use_device_memory : decoder output surface is in device memory if true else on host memory
    )pbdoc"
                )
        ;
   
    ExternalBuffer::Export(m);

    py::class_<DecodedFrame, std::shared_ptr<DecodedFrame>>(m, "DecodedFrame")
        .def_readonly("timestamp", &DecodedFrame::timestamp)
        .def_readonly("format", &DecodedFrame::format)
        .def("__repr__",
            [](std::shared_ptr<DecodedFrame>& self)
            {
                std::stringstream ss;
                ss << "<DecodedFrame [";
                ss << "timestamp=" << self->timestamp;
                ss << ", format=" << py::str(py::cast(self->format));
                ss << ", " << py::str(py::cast(self->views));
                ss << "]>";
                return ss.str();
            })
        .def("framesize",
            [](std::shared_ptr<DecodedFrame>& self) {
                int height = self->views.at(0).shape.at(0);
                int width = self->views.at(0).shape.at(1);
                int framesize = width * height * 1.5;
                switch (self->format)
                {
                case Pixel_Format_NV12:
                        break;
                case Pixel_Format_P016:
                    framesize = width * height * 3;
                    break;
                case Pixel_Format_YUV444:
                    framesize = width * height * 3;
                    break;
                case Pixel_Format_YUV444_16Bit:
                    framesize = width * height * 6;
                    break;
                default:
                    break;
                }
                return framesize;
            },
            R"pbdoc(
            return underlying views which implement CAI
            :param None: None
            )pbdoc")
        .def("cuda",
            [](std::shared_ptr<DecodedFrame>& self) {
                return self->views;
            },
            R"pbdoc(
            return underlying views which implement CAI
            :param None: None
            )pbdoc")
         .def("nvcv_image",
             [](std::shared_ptr<DecodedFrame>& self) {
                 switch (self->format)
                 {
                     case Pixel_Format_NV12:
                     {
                         
                         size_t width = self->views.at(0).shape[1];
                         size_t height = self->views.at(0).shape[0] * 1.5;
                         CUdeviceptr data = self->views.at(0).data;
                         CUstream stream = self->views.at(0).stream;
                         self->views.clear();
                         self->views.push_back(CAIMemoryView{ { height, width, 1}, {width, 2, 1}, "|u1", reinterpret_cast<size_t>(stream),(data), false }); //hack for cvcuda tensor represenation
                     }
                     break;
                     case Pixel_Format_YUV444:
                     {

                         size_t width = self->views.at(0).shape[1];
                         size_t height = self->views.at(0).shape[0] * 3;
                         CUdeviceptr data = self->views.at(0).data;
                         CUstream stream = self->views.at(0).stream;
                         self->views.clear();
                         self->views.push_back(CAIMemoryView{ { height, width, 1}, {width, 3, 1}, "|u1", reinterpret_cast<size_t>(stream),(data), false }); //hack for cvcuda tensor represenation
                     }
                break;
             default:
                throw std::invalid_argument("only nv12 and yuv444 supported as of now");
                break;
                 }
                 return self->views;
             },
             R"pbdoc(
            return underlying views which implement CAI
            :param None: None
            )pbdoc")

            // DL Pack Tensor
             .def_property_readonly("shape", [](std::shared_ptr<DecodedFrame>& self) {
                 return self->extBuf->shape();
                    }, "Get the shape of the buffer as an array")
             .def_property_readonly("strides", [](std::shared_ptr<DecodedFrame>& self) {
                        return self->extBuf->strides();
                 }, "Get the strides of the buffer")
             .def_property_readonly("dtype", [](std::shared_ptr<DecodedFrame>& self) {
                     return self->extBuf->dtype();
                 }, "Get the data type of the buffer")
             .def("__dlpack__", [](std::shared_ptr<DecodedFrame>& self, py::object stream) {
                 return self->extBuf->dlpack(stream);
                    }, py::arg("stream") = NULL, "Export the buffer as a DLPack tensor")
             .def("__dlpack_device__", [](std::shared_ptr<DecodedFrame>& self) {
                        //DLDevice ctx;
                        //ctx.device_type = DLDeviceType::kDLCUDA;
                        //ctx.device_id = 0;
                        return py::make_tuple(py::int_(static_cast<int>(DLDeviceType::kDLCUDA)),
                               py::int_(static_cast<int>(0)));
                 }, "Get the device associated with the buffer")
            
            
            .def("GetPtrToPlane",

                [](std::shared_ptr<DecodedFrame>& self, int planeIdx) {
                    return self->views[planeIdx].data;
                    }, R"pbdoc(
            return pointer to base address for plane index
            :param planeIdx : index to the plane
            )pbdoc");
            // TODO add __iter__ interface on DecodedFrame
            py::class_<CAIMemoryView, std::shared_ptr<CAIMemoryView>>(m, "CAIMemoryView")
                .def(py::init<std::vector<size_t>, std::vector<size_t>, std::string, size_t, CUdeviceptr, bool>())
                .def_readonly("shape", &CAIMemoryView::shape)
                .def_readonly("stride", &CAIMemoryView::stride)
                .def_readonly("dataptr", &CAIMemoryView::data)
                .def("__repr__",
                    [](std::shared_ptr<CAIMemoryView>& self)
                    {
                        std::stringstream ss;
                        ss << "<CAIMemoryView ";
                        ss << py::str(py::cast(self->shape));
                        ss << ">";
                        return ss.str();
                    })
                .def_readonly("data", &CAIMemoryView::data)
                        .def_property_readonly("__cuda_array_interface__",
                            [](std::shared_ptr<CAIMemoryView>& self)
                            {
                                py::dict dict;
                                dict["version"] = 3;
                                dict["shape"] = self->shape;
                                dict["strides"] = self->stride;
                                dict["typestr"] = self->typestr;
                                dict["stream"] = self->stream == 0 ? int(size_t(self->stream)) : 2;
                                dict["data"] = std::make_pair(self->data, false);
                                dict["gpuIdx"] = 0;  // TODO
                                return dict;
                            });


                    py::class_<PyNvDecoder, shared_ptr<PyNvDecoder>>(m, "PyNvDecoder", py::module_local())
                        .def(py::init<>(),
                            R"pbdoc(
        Constructor method. Initialize decoder with set of particular
        parameters
        :param None: None
    )pbdoc")
                                .def("setDecoderSessionID",
                                    [](std::shared_ptr<PyNvDecoder>& dec, int sessionID)
                                    {
                                        dec->setDecoderSessionID(sessionID);
                                        
                                    })
                                .def_static("getDecoderSessionOverHead",
                                    [](int sessionID)
                                    {
                                       return PyNvDecoder::getDecoderSessionOverHead(sessionID);
                                    })
                                .def("GetPixelFormat",
                                    [](std::shared_ptr<PyNvDecoder>& dec)
                                    {
                                        return dec->GetNativeFormat(dec->GetOutputFormat());
                                    },R"pbdoc(
            Returns Pixel format string representation 
            :param None
            :return: String representation of pixel format is returned
    )pbdoc")
                                        .def(
                                            "GetNumDecodedFrame",
                                            [](std::shared_ptr<PyNvDecoder>& dec, const PacketData& packetData)
                                            {
                                                return dec->GetNumDecodedFrame(packetData);
                                            }, R"pbdoc(
             Decodes bistream data in Packet into uncompressed data 
            :param PacketData: PacketData Structure
            :return: count of the decoded frames
    )pbdoc"
                                                )
                                        .def(
                                            "GetLockedFrame",
                                            [](std::shared_ptr<PyNvDecoder>& dec)
                                            {
                                                uint8_t* pFrame = dec->GetLockedFrame(NULL);
                                                return (CUdeviceptr)pFrame;
                                            }, R"pbdoc(
            This function decodes a frame and returns the locked frame buffers
            This makes the buffers available for use by the application without the buffers
            getting overwritten, even if subsequent decode calls are made. The frame buffers
            remain locked, until UnlockFrame() is called
            :param NULL
            :return: returns decodedFrame from internal buffer
    )pbdoc"
                                                )
                                        .def(
                                            "UnlockFrame",
                                            [](std::shared_ptr<PyNvDecoder>& dec, CUdeviceptr pFrame)
                                            {
                                                return dec->UnlockFrame((uint8_t*)pFrame);
                                            }, R"pbdoc(
            This function unlocks the frame buffer and makes the frame buffers available for write again
            :param NULL
            :return: void
    )pbdoc"
                                                )
                                        .def(
                                            "Decode",
                                            [](std::shared_ptr<PyNvDecoder>& dec, const PacketData& packetData)
                                            {
                                                return dec->Decode(packetData);
                                            }, R"pbdoc(
            Decodes bistream data in Packet into uncompressed data 
            :param PacketData: PacketData Structure
            :return: uncompressed data is returned as List of Decoded Frames
    )pbdoc"
                                                )
                                .def(
                                    "Decode",
                                    [](std::shared_ptr<PyNvDecoder>& dec, const PacketData& packetData)
                                    {
                                        return dec->Decode(packetData);
                                    }, R"pbdoc(
            Decodes bistream data in Packet into uncompressed data 
            :param PacketData: PacketData Structure
            :return: uncompressed data is returned as List of Decoded Frames
    )pbdoc"
                                        )
				.def(
				"GetWidth",
				[](std::shared_ptr<PyNvDecoder>& dec)
				{
					return dec->GetWidth();
				},R"pbdoc()pbdoc"
				"Get the width of decoded frame"
				)
				.def(
				"GetHeight",
				[](std::shared_ptr<PyNvDecoder>& dec)
				{
					return dec->GetHeight();
				},R"pbdoc()pbdoc"
				"Get the height of decoded frame"
				)
				.def(
				"GetFrameSize",
				[](std::shared_ptr<PyNvDecoder>& dec)
				{
					return dec->GetFrameSize();
				},R"pbdoc()pbdoc"
				"Get the size of decoded frame"
				)
 
                                        .def(
                                            "WaitOnCUStream",
                                            [](std::shared_ptr<PyNvDecoder>& dec, size_t _stream)
                                            {
                                                return dec->CUStreamWaitOnEvent(reinterpret_cast<CUstream>(_stream));
                                            }, R"pbdoc(
           Wait for post proc kernels + memcopy to finish so that input stream can access it
            :param application created CUDA Stream
            :return: None
    )pbdoc"
                                           )
                                        .def(
                                            "SyncOnCUStream",
                                            [](std::shared_ptr<PyNvDecoder>& dec)
                                            {
                                                return dec->CUStreamSyncOnEvent();
                                            }, R"pbdoc(
           Sync forces post proc kernels + memcopy to finish so that any stream can access it
            :param None
            :return: None
    )pbdoc"
                                                )
                                        .def(
                                            "__iter__",
                                            [](shared_ptr<PyNvDecoder> self) {
                                                return self;
                                            },
                                            R"pbdoc(
                                        Iterator over decoder object
    )pbdoc")
                                        .def(
                                            "__next__",
                                            [](shared_ptr<PyNvDecoder> self, const PacketData& packetData) {

                                                if (packetData.bsl != 0)
                                                {
                                                    return self->Decode(packetData);
                                                }
                                                else
                                                {
                                                    throw py::stop_iteration();
                                                }

                                            },
                                            R"pbdoc(
            gets the next element in Iterator over decoder object
    )pbdoc");
}
