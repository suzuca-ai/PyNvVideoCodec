// SPDX-FileCopyrightText: Copyright (c) 2026 Suzuca Technologies
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
}

// Reuse FFmpeg's framing and PTS tracking; metadata is not exposed by its parser.
class HevcAccessUnits {
    AVCodecParserContext* parser = nullptr;
    AVCodecContext* codec = nullptr;
    size_t pending = 0;
    static constexpr size_t maxAccessUnit = 64 * 1024 * 1024;
public:
    HevcAccessUnits() = default;
    HevcAccessUnits(const HevcAccessUnits&) = delete;
    HevcAccessUnits& operator=(const HevcAccessUnits&) = delete;
    ~HevcAccessUnits() { reset(); }
    void reset() {
        if (parser) av_parser_close(parser);
        parser = nullptr;
        avcodec_free_context(&codec);
        pending = 0;
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
                    pts, AV_NOPTS_VALUE, parser->cur_offset);
                pts = AV_NOPTS_VALUE;
                // After an explicit frame flush, FFmpeg can first report an
                // empty boundary while clearing its previous frame-start state.
                const bool noProgress = !consumed && !outputSize && count;
                if (consumed < 0 || (noProgress && stalled))
                    throw std::runtime_error("Cannot frame HEVC access unit");
                stalled = noProgress;
                pending += consumed;
                if (pending > maxAccessUnit || static_cast<size_t>(outputSize) > maxAccessUnit)
                    throw std::runtime_error("HEVC access unit exceeds 64 MiB");
                if (outputSize) { pending = 0; emit(output, outputSize, parser->pts); }
                offset += consumed;
            } while (offset < count);
            if (count) data += count;
            size -= count;
            if (!count) break;
        }
    }
};
