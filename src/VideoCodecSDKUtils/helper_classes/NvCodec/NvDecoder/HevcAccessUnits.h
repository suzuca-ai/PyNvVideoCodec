// SPDX-FileCopyrightText: Copyright (c) 2026 Suzuca Technologies
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
}

// Reuse FFmpeg framing; match timestamps to the first VCL, not preceding SPS/PPS.
class HevcAccessUnits {
    AVCodecParserContext* parser = nullptr;
    AVCodecContext* codec = nullptr;
    size_t pending = 0;
    uint64_t header = 0;
    std::array<int64_t, 7> headerPts{};
    unsigned head = 0;
    bool pictureSeen = false;
    int64_t picturePts = AV_NOPTS_VALUE, leadingPts = AV_NOPTS_VALUE;

    void track(const uint8_t* data, size_t size, int64_t pts, bool& used) {
        for (size_t i = 0; i < size; ++i) {
            header = (header << 8) | data[i];
            headerPts[head] = used ? AV_NOPTS_VALUE : pts;
            if (!used && leadingPts == AV_NOPTS_VALUE) leadingPts = pts;
            head = (head + 1) % headerPts.size();
            if (((header >> 24) & 0xffffff) != 1 || ((header >> 17) & 63) > 31 || !(header & 128))
                continue;
            if (!pictureSeen) {
                pictureSeen = true;
                const unsigned bytes = ((header >> 48) & 255) ? 6 : 7;
                for (unsigned j = headerPts.size() - bytes; j < headerPts.size(); ++j) {
                    const auto index = (head + j) % headerPts.size();
                    if (headerPts[index] != AV_NOPTS_VALUE) picturePts = headerPts[index];
                }
            }
            used = true; // One input packet timestamps only its first picture.
        }
    }
    static constexpr size_t maxAccessUnit = 64 * 1024 * 1024;
public:
    HevcAccessUnits() { headerPts.fill(AV_NOPTS_VALUE); }
    HevcAccessUnits(const HevcAccessUnits&) = delete;
    HevcAccessUnits& operator=(const HevcAccessUnits&) = delete;
    ~HevcAccessUnits() { reset(); }
    void reset() {
        if (parser) av_parser_close(parser);
        parser = nullptr;
        avcodec_free_context(&codec);
        pending = 0;
        header = 0; head = 0;
        headerPts.fill(AV_NOPTS_VALUE);
        pictureSeen = false;
        picturePts = leadingPts = AV_NOPTS_VALUE;
    }
    // emit() must consume the borrowed access unit before it returns.
    template<class Emit>
    void feed(const uint8_t* data, size_t size, int64_t pts, bool flush, Emit emit) {
        if (!parser) {
            parser = av_parser_init(AV_CODEC_ID_HEVC);
            codec = avcodec_alloc_context3(nullptr);
            if (!parser || !codec) { reset(); throw std::runtime_error("Cannot initialize HEVC access-unit parser"); }
            codec->codec_id = AV_CODEC_ID_HEVC;
        }
        bool timestampUsed = false;
        std::vector<uint8_t> padded;
        while (size || flush) {
            const size_t count = std::min(size, size_t{65536});
            padded.assign(count + AV_INPUT_BUFFER_PADDING_SIZE, 0);
            if (count) std::copy_n(data, count, padded.data());
            size_t offset = 0;
            bool stalled = false;
            do {
                uint8_t* output = nullptr;
                int outputSize = 0;
                const int consumed = av_parser_parse2(parser, codec, &output, &outputSize,
                    count ? padded.data() + offset : nullptr, static_cast<int>(count - offset),
                    AV_NOPTS_VALUE, AV_NOPTS_VALUE, -1);
                // After an explicit frame flush, FFmpeg can first report an
                // empty boundary while clearing its previous frame-start state.
                const bool noProgress = !consumed && !outputSize && count;
                if (consumed < 0 || (noProgress && stalled))
                    throw std::runtime_error("Cannot frame HEVC access unit");
                stalled = noProgress;
                pending += consumed;
                if (static_cast<size_t>(outputSize) > pending || static_cast<size_t>(outputSize) > maxAccessUnit ||
                    pending - outputSize > maxAccessUnit)
                    throw std::runtime_error("HEVC access unit exceeds 64 MiB");
                // Follow only consumed bytes: next-AU lookahead must not take
                // this picture's timestamp. Header state survives AU boundaries.
                track(padded.data() + offset, consumed, pts, timestampUsed);
                if (outputSize) {
                    pending -= outputSize;
                    emit(output, outputSize, picturePts != AV_NOPTS_VALUE ? picturePts : leadingPts);
                    pictureSeen = false;
                    picturePts = leadingPts = AV_NOPTS_VALUE;
                    // A start code split across packets may already have been
                    // consumed. Carry only those bounded lookahead bytes' PTS.
                    for (size_t j = headerPts.size() - std::min(pending, headerPts.size()); j < headerPts.size(); ++j)
                        if (headerPts[(head + j) % headerPts.size()] != AV_NOPTS_VALUE)
                            leadingPts = headerPts[(head + j) % headerPts.size()];
                }
                offset += consumed;
            } while (offset < count);
            if (count) data += count;
            size -= count;
            if (!count) break;
        }
    }
};
