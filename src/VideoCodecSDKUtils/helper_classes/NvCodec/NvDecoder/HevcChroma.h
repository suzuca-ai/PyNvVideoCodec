// SPDX-FileCopyrightText: Copyright (c) 2026 Suzuca Technologies
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
extern "C" {
#include <libavcodec/bsf.h>
}

// HEVC Annex B metadata only. Values match AVChromaLocation: unknown=0,
// left=1, center=2, top-left=3, top=4, bottom-left=5, bottom=6.
// The decoder still owns bitstream validation and picture decoding.
class HevcChroma {
    struct Bits {
        const uint8_t* data;
        size_t size, offset = 2;
        unsigned remaining = 0, byte = 0, zeros = 0;
        bool valid = true;
        Bits(const uint8_t* data, size_t size) : data(data), size(size) {}
        explicit Bits(const std::vector<uint8_t>& nal) : Bits(nal.data(), nal.size()) {}
        uint32_t read(unsigned count) {
            if (count > 32) { valid = false; return 0; }
            uint32_t result = 0;
            while (count-- && valid) {
                if (!remaining) {
                    if (offset == size) { valid = false; return 0; }
                    byte = data[offset++];
                    if (zeros == 2 && byte == 3) {
                        zeros = 0;
                        if (offset == size) { valid = false; return 0; }
                        byte = data[offset++];
                    }
                    zeros = byte == 0 ? zeros + 1 : 0;
                    remaining = 8;
                }
                result = (result << 1) | ((byte >> --remaining) & 1);
            }
            return result;
        }
        void skip(unsigned count) { while (count && valid) { const auto n = std::min(count, 32U); read(n); count -= n; } }
        uint32_t ue(uint32_t maximum = 65535) {
            unsigned zeros = 0;
            while (valid && !read(1)) {
                if (++zeros > 31) { valid = false; return 0; }
            }
            const uint32_t result = ((uint32_t{1} << zeros) - 1) + read(zeros);
            if (!valid || result > maximum) { valid = false; return 0; }
            return result;
        }
    };

    std::array<uint8_t, 16> sps{};
    std::array<int, 64> pps;
    std::vector<uint8_t> nal;
    AVBSFContext* validator = nullptr;
    bool oversized = false, inNal = false;
    uint8_t tail[2] = {255, 255};

    void collect(const uint8_t* data, size_t size) {
        const auto prefix = std::min(size, 2 - std::min(nal.size(), size_t{2}));
        nal.insert(nal.end(), data, data + prefix);
        data += prefix; size -= prefix;
        if (nal.size() < 2) return;
        const unsigned type = (nal[0] >> 1) & 63;
        if (type != 33 && type != 34) return;
        const auto count = std::min(size, 65536 - nal.size());
        nal.insert(nal.end(), data, data + count);
        oversized |= count != size;
    }


    bool validParameterSet() {
        if (oversized) return false;
        if (!validator) {
            const auto* filter = av_bsf_get_by_name("hevc_metadata");
            if (!filter || av_bsf_alloc(filter, &validator) < 0) return false;
            validator->par_in->codec_id = AV_CODEC_ID_HEVC;
            validator->par_in->codec_type = AVMEDIA_TYPE_VIDEO;
            if (av_bsf_init(validator) < 0) { av_bsf_free(&validator); return false; }
        }
        AVPacket* packet = av_packet_alloc();
        if (!packet) return false;
        bool valid = false;
        if (av_new_packet(packet, static_cast<int>(nal.size() + 3)) >= 0) {
            packet->data[0] = packet->data[1] = 0; packet->data[2] = 1;
            std::copy(nal.begin(), nal.end(), packet->data + 3);
            // CBS validates the complete SPS/PPS, including extensions and
            // RBSP termination. Never validate or copy picture data here.
            valid = av_bsf_send_packet(validator, packet) >= 0 &&
                    av_bsf_receive_packet(validator, packet) >= 0;
        }
        av_packet_free(&packet);
        return valid;
    }

    void readParameterSet() {
        if (nal.size() < 3 || (nal[0] & 0x81) || (nal[1] & 0xf8) || !(nal[1] & 7)) return;
        const unsigned type = (nal[0] >> 1) & 63;
        if (type != 33 && type != 34) return;
        Bits b(nal);
        if (type == 34) {
            const auto id = b.ue(63);
            if (!b.valid) return;
            pps[id] = -1;
            const auto source = b.ue(15);
            if (b.valid && validParameterSet()) pps[id] = static_cast<int>(source);
            return;
        }
        b.skip(4);
        const auto layers = b.read(3);
        b.skip(1);
        b.skip(96); // general profile_tier_level
        std::array<bool, 7> profile{}, level{};
        for (unsigned i = 0; i < layers; ++i) { profile[i] = b.read(1); level[i] = b.read(1); }
        if (layers) b.skip((8 - layers) * 2);
        for (unsigned i = 0; i < layers; ++i) {
            if (profile[i]) b.skip(88);
            if (level[i]) b.skip(8);
        }
        const auto id = b.ue(15);
        if (!b.valid || layers > 6) return;
        sps[id] = 0; // A malformed replacement must not inherit the previous VUI.
        if (!validParameterSet()) return;
        const auto chroma = b.ue(3);
        if (chroma == 3) b.skip(1);
        b.ue(); b.ue();
        if (b.read(1)) for (int i = 0; i < 4; ++i) b.ue();
        b.ue(8); b.ue(8);
        const auto pocBits = b.ue(12) + 4;
        const auto firstLayer = b.read(1) ? 0 : layers;
        for (unsigned i = firstLayer; i <= layers; ++i) { b.ue(16); b.ue(16); b.ue(); }
        for (int i = 0; i < 6; ++i) b.ue(32);
        if (b.read(1) && b.read(1)) { // scaling_list_data
            for (unsigned size = 0; size < 4; ++size) {
                for (unsigned matrix = 0; matrix < 6; matrix += size == 3 ? 3 : 1) {
                    if (!b.read(1)) b.ue(6);
                    else {
                        if (size > 1) b.ue(510); // signed Exp-Golomb; only its length matters
                        for (unsigned i = 0; i < std::min(64U, 1U << (4 + 2 * size)); ++i) b.ue(510);
                    }
                }
            }
        }
        b.skip(2); // amp_enabled_flag, sample_adaptive_offset_enabled_flag
        if (b.read(1)) { b.skip(8); b.ue(32); b.ue(32); b.skip(1); }
        const auto sets = b.ue(64);
        unsigned previous = 0;
        for (unsigned i = 0; i < sets && b.valid; ++i) {
            unsigned count = 0;
            if (i && b.read(1)) {
                b.skip(1); b.ue();
                for (unsigned j = 0; j <= previous; ++j) {
                    const bool used = b.read(1);
                    if (used || b.read(1)) ++count;
                }
            } else {
                const auto negative = b.ue(16), positive = b.ue(16);
                count = negative + positive;
                for (unsigned j = 0; j < count; ++j) { b.ue(); b.skip(1); }
            }
            if (count > 16) b.valid = false;
            previous = count;
        }
        if (b.read(1)) {
            const auto count = b.ue(32);
            for (unsigned i = 0; i < count; ++i) b.skip(pocBits + 1);
        }
        b.skip(2); // temporal MVP, strong intra smoothing
        uint8_t location = 1; // HEVC infers type 0 when chroma_loc_info is absent.
        if (b.read(1)) { // vui_parameters_present_flag
            if (b.read(1) && b.read(8) == 255) b.skip(32);
            if (b.read(1)) b.skip(1);
            if (b.read(1)) { b.skip(4); if (b.read(1)) b.skip(24); }
            if (b.read(1)) {
                const auto top = b.ue(5), bottom = b.ue(5);
                location = top == bottom ? static_cast<uint8_t>(top + 1) : 0;
            }
        }
        if (b.valid && chroma == 1) sps[id] = location;
    }

public:
    HevcChroma() { pps.fill(-1); }
    HevcChroma(const HevcChroma&) = delete;
    HevcChroma& operator=(const HevcChroma&) = delete;
    ~HevcChroma() { av_bsf_free(&validator); }
    void reset() {
        sps.fill(0); pps.fill(-1); nal.clear(); oversized = inNal = false;
        tail[0] = tail[1] = 255;
        av_bsf_free(&validator);
    }

    // NVDEC supplies the first slice; only its PPS id is read, without allocation.
    uint8_t pictureLocation(const uint8_t* data, size_t size) const {
        size = std::min(size, size_t{32});
        while (size && !*data) { ++data; --size; }
        if (size && *data == 1) { ++data; --size; }
        if (size < 3 || (data[0] & 0x81) || (data[1] & 0xf8) || !(data[1] & 7)) return 0;
        const unsigned type = (data[0] >> 1) & 63;
        if (type > 31) return 0;
        Bits b(data, size);
        if (!b.read(1)) return 0;
        if (type >= 16 && type <= 23) b.skip(1);
        const auto id = b.ue(63);
        return b.valid && pps[id] >= 0 ? sps[pps[id]] : 0;
    }

    // Forward borrowed input to NVDEC. Only SPS/PPS are buffered (64 KiB max).
    // Split at parameter-set updates so prior decode callbacks see the old cache.
    // Picture framing, reorder and timestamps remain owned by NVDEC.
    template<class Forward>
    void feed(const uint8_t* data, size_t size, bool flush, bool completePicture, Forward forward) {
        size_t begin = 0, sent = 0, search = 0;
        while (search < size) {
            const auto* found = static_cast<const uint8_t*>(std::memchr(data + search, 1, size - search));
            if (!found) break;
            const size_t i = found - data;
            search = i + 1;
            if ((i ? data[i - 1] : tail[1]) || (i >= 2 ? data[i - 2] : tail[i])) continue;
            if (inNal) collect(data + begin, i - begin);
            if (nal.size() >= 2 && (((nal[0] >> 1) & 63) == 33 || ((nal[0] >> 1) & 63) == 34)) {
                forward(data + sent, i + 1 - sent, false);
                sent = i + 1;
                while (!nal.empty() && !nal.back()) nal.pop_back();
                readParameterSet();
            }
            nal.clear(); oversized = false; inNal = true;
            begin = i + 1;
            // ENDOFPICTURE guarantees one picture. Parameter sets precede VCL;
            // skip compressed slice payload entirely in this common packet mode.
            if (completePicture && begin + 2 <= size && ((data[begin] >> 1) & 63) <= 31) break;
        }
        if (inNal && size > begin) collect(data + begin, size - begin);
        if (size >= 2) { tail[0] = data[size - 2]; tail[1] = data[size - 1]; }
        else if (size) { tail[0] = tail[1]; tail[1] = data[0]; }
        forward(data ? data + sent : nullptr, size - sent, true);
        if (flush) {
            while (!nal.empty() && !nal.back()) nal.pop_back();
            readParameterSet();
            nal.clear(); oversized = inNal = false; tail[0] = tail[1] = 255;
        }
    }
};
