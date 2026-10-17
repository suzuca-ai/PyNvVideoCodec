// SPDX-FileCopyrightText: Copyright (c) 2026 Suzuca Technologies
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>
#include <utility>

// HEVC Annex B metadata only. Values match AVChromaLocation: unknown=0,
// left=1, center=2, top-left=3, top=4, bottom-left=5, bottom=6.
// The decoder still owns bitstream validation and picture decoding.
class HevcChroma {
    struct Bits {
        std::vector<uint8_t> rbsp;
        size_t position = 0;
        bool valid = true;
        explicit Bits(const std::vector<uint8_t>& nal) {
            unsigned zeros = 0;
            for (size_t i = 2; i < nal.size(); ++i) {
                const auto byte = nal[i];
                if (zeros == 2 && byte == 3) { zeros = 0; continue; }
                rbsp.push_back(byte);
                zeros = byte == 0 ? zeros + 1 : 0;
            }
        }
        uint32_t read(unsigned count) {
            if (count > 32 || position + count > rbsp.size() * 8) {
                valid = false;
                return 0;
            }
            uint32_t result = 0;
            while (count--) {
                result = (result << 1) | ((rbsp[position / 8] >> (7 - position % 8)) & 1);
                ++position;
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
            if (result > maximum) valid = false;
            return result;
        }
    };

    struct Picture {
        std::vector<uint8_t> prefix;
        uint8_t location;
    };
    std::array<uint8_t, 16> sps{};
    std::array<int, 64> pps;
    std::deque<std::shared_ptr<Picture>> pictures;
    std::shared_ptr<Picture> currentPicture;
    std::vector<uint8_t> nal;
    unsigned zeros = 0;
    bool inNal = false;

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
            if (b.valid) pps[id] = static_cast<int>(source);
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

    void queuePicture() {
        if (currentPicture) {
            currentPicture->prefix.assign(nal.begin(), nal.begin() + std::min(nal.size(), size_t{32}));
            return;
        }
        if (nal.size() < 3 || ((nal[0] >> 1) & 63) > 31) return;
        Bits b(nal);
        if (!b.read(1)) return; // Only the first slice identifies a new picture.
        const unsigned type = (nal[0] >> 1) & 63;
        if (type >= 16 && type <= 23) b.skip(1);
        const auto id = b.ue(63);
        if (!b.valid) return;
        uint8_t location = 0;
        if (!(nal[0] & 0x81) && !(nal[1] & 0xf8) && (nal[1] & 7) && pps[id] >= 0)
            location = sps[pps[id]];
        currentPicture = std::make_shared<Picture>(Picture{
            std::vector<uint8_t>(nal.begin(), nal.begin() + std::min(nal.size(), size_t{32})), location});
        pictures.push_back(currentPicture);
    }

    void finishNal() {
        readParameterSet();
        queuePicture();
        nal.clear();
        currentPicture.reset();
    }

public:
    HevcChroma() { pps.fill(-1); }
    void reset() { *this = HevcChroma(); }

    // Inspect before cuvidParseVideoData; a single packet may contain several
    // parameter-set replacements. Retain only parameter sets / slice prefixes.
    void feed(const uint8_t* data, size_t size, bool end = false) {
        for (size_t i = 0; i < size; ++i) {
            const auto byte = data[i];
            if (!byte) { zeros = std::min(zeros + 1, 65536U); continue; }
            if (byte == 1 && zeros >= 2) {
                finishNal();
                inNal = true;
                zeros = 0;
                continue;
            }
            if (inNal) {
                const size_t limit = nal.size() >= 2 && ((nal[0] >> 1) & 63) <= 31 ? 32 : 65536;
                while (zeros && nal.size() < limit) { nal.push_back(0); --zeros; }
                if (nal.size() < limit) nal.push_back(byte);
            }
            zeros = 0;
        }
        // Packet boundaries are not NAL boundaries; SPS/PPS wait for a start
        // code. A complete first-slice prefix is enough for a decode callback.
        queuePicture();
        if (end) { finishNal(); inNal = false; zeros = 0; }
    }

    uint8_t take(const uint8_t* data, size_t size) {
        // NVDEC's picture bitstream retains Annex B start codes. Match the first
        // slice too, so a rejected/skipped picture cannot shift later metadata.
        size_t offset = 0;
        while (offset < size && data[offset] == 0) ++offset;
        if (offset >= 2 && offset < size && data[offset] == 1) ++offset;
        else offset = 0;
        while (!pictures.empty()) {
            const auto picture = std::move(pictures.front());
            pictures.pop_front();
            if (size - offset >= picture->prefix.size() &&
                std::equal(picture->prefix.begin(), picture->prefix.end(), data + offset)) return picture->location;
        }
        return 0;
    }
};
