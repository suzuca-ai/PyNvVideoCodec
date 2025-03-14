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
#include "NvDemuxer.hpp"

using namespace std;
using namespace chrono;

namespace py = pybind11;

#ifdef DEMUX_ONLY
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();
#endif

NvDemuxer::NvDemuxer(const std::string& inputfile)
{
    demuxer.reset(new FFmpegDemuxer(inputfile.c_str()));
    currentPacket.reset(new PacketData());
    isEOSReached = false;
}

shared_ptr<PacketData> NvDemuxer::Demux()
{
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = NULL, * pFrame;
    memset(currentPacket.get(), 0, sizeof(PacketData));

    if (demuxer->Demux(&pVideo, &nVideoBytes))
    {
        if (nVideoBytes)
        {
            currentPacket.get()->bsl_data = (uintptr_t)pVideo;
            currentPacket.get()->bsl = nVideoBytes;
        }
    }
    else
    {
        isEOSReached = true;
    }
    return currentPacket;
}

shared_ptr<PacketData> NvDemuxer:: Seek(uint64_t timestamp)
{
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pVideo = NULL, * pFrame;
    SeekContext ctx;
    ctx.seek_frame = timestamp;
    if (demuxer->Seek(ctx, &pVideo, &nVideoBytes))
    {
        currentPacket.get()->bsl_data = (uintptr_t)pVideo;
        currentPacket.get()->bsl = nVideoBytes;
    }
    return currentPacket;
}

ColorSpace NvDemuxer::GetColorSpace() const
{
    switch (demuxer->GetColorSpace()) {
    case AVCOL_SPC_BT709:
        return ColorSpace::BT_709;
        break;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
        return ColorSpace::BT_601;
        break;
    default:
        return ColorSpace::UNSPEC;
        break;
    }
}

ColorRange NvDemuxer::GetColorRange() const
{
    switch (demuxer->GetColorRange()) {
    case AVCOL_RANGE_MPEG:
        return ColorRange::MPEG;
        break;
    case AVCOL_RANGE_JPEG:
        return ColorRange::JPEG;
        break;
    default:
        return ColorRange::UDEF;
        break;
    }
}
