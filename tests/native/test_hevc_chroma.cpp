// SPDX-FileCopyrightText: Copyright (c) 2026 Suzuca Technologies
// SPDX-License-Identifier: MIT
#include "HevcChroma.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>

struct Writer {
    std::vector<uint8_t> bytes;
    unsigned bit = 0;
    void put(uint32_t value, unsigned count = 1) {
        while (count--) {
            if (!(bit % 8)) bytes.push_back(0);
            bytes.back() |= ((value >> count) & 1) << (7 - bit++ % 8);
        }
    }
    void ue(uint32_t value) {
        unsigned width = 0;
        for (auto n = value + 1; n; n >>= 1) ++width;
        put(0, width - 1); put(value + 1, width);
    }
    std::vector<uint8_t> nal(unsigned type) {
        put(1);
        std::vector<uint8_t> out = {0, 0, 1, static_cast<uint8_t>(type << 1), 1};
        unsigned zeros = 0;
        for (auto byte : bytes) {
            if (zeros == 2 && byte <= 3) { out.push_back(3); zeros = 0; }
            out.push_back(byte);
            zeros = byte == 0 ? zeros + 1 : 0;
        }
        return out;
    }
};

static std::vector<uint8_t> sps(int top, int bottom = -2, bool complex = false) {
    Writer w;
    w.put(0, 4); w.put(0, 3); w.put(1);
    for (int i = 0; i < 3; ++i) w.put(0, 32);
    w.ue(0); w.ue(1); w.ue(384); w.ue(256); w.put(0);
    w.ue(0); w.ue(0); w.ue(4); w.put(0);
    w.ue(4); w.ue(2); w.ue(0);
    for (int i = 0; i < 6; ++i) w.ue(0);
    w.put(complex);
    if (complex) {
        w.put(1);
        for (unsigned size = 0; size < 4; ++size)
            for (unsigned matrix = 0; matrix < 6; matrix += size == 3 ? 3 : 1) { w.put(0); w.ue(0); }
    }
    w.put(1); w.put(1); w.put(complex);
    if (complex) { w.put(7, 4); w.put(7, 4); w.ue(0); w.ue(0); w.put(0); }
    w.ue(complex ? 2 : 0);
    if (complex) {
        w.ue(1); w.ue(0); w.ue(0); w.put(1);
        w.put(1); w.put(0); w.ue(0); w.put(1); w.put(0); w.put(1);
    }
    w.put(complex);
    if (complex) { w.ue(1); w.put(5, 8); w.put(1); }
    w.put(1); w.put(1);
    w.put(top != -2);
    if (top != -2) {
        w.put(complex);
        if (complex) { w.put(255, 8); w.put(1, 16); w.put(1, 16); }
        w.put(complex); if (complex) w.put(0);
        w.put(complex);
        if (complex) { w.put(5, 3); w.put(0); w.put(1); w.put(1, 8); w.put(1, 8); w.put(1, 8); }
        w.put(top >= 0);
        if (top >= 0) { w.ue(top); w.ue(bottom == -2 ? top : bottom); }
    }
    return w.nal(33);
}

static std::vector<uint8_t> pps() { Writer w; w.ue(0); w.ue(0); return w.nal(34); }
static std::vector<uint8_t> slice(unsigned id, bool first = true) {
    Writer w; w.put(first); w.ue(0); w.put(id, 16); return w.nal(1);
}
static void append(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src) { dst.insert(dst.end(), src.begin(), src.end()); }

static void unitTests() {
    for (bool complex : {false, true}) {
        std::vector<uint8_t> stream;
        std::vector<std::vector<uint8_t>> frames;
        std::vector<int> expected;
        for (int value : {-2, -1, 0, 1, 2, 3, 4, 5, 6}) {
            append(stream, sps(value, -2, complex)); append(stream, pps());
            frames.push_back(slice(frames.size())); append(stream, frames.back());
            append(stream, slice(999, false)); // Additional slices do not consume metadata.
            expected.push_back(value < 0 ? 1 : value <= 5 ? value + 1 : 0);
        }
        append(stream, sps(0, 1)); append(stream, pps());
        frames.push_back(slice(100)); append(stream, frames.back()); expected.push_back(0);
        for (size_t chunk : {size_t{1}, size_t{7}, size_t{733}, stream.size()}) {
            HevcChroma parser;
            for (size_t i = 0; i < stream.size(); i += chunk)
                parser.feed(stream.data() + i, std::min(chunk, stream.size() - i));
            parser.feed(nullptr, 0, true);
            for (size_t i = 0; i < frames.size(); ++i)
                assert(parser.take(frames[i].data(), frames[i].size()) == expected[i]);
            assert(parser.take(frames[0].data(), frames[0].size()) == 0);
        }
    }
    HevcChroma parser;
    std::vector<uint8_t> stream;
    append(stream, sps(2)); append(stream, pps()); append(stream, slice(1));
    append(stream, sps(5)); append(stream, pps()); append(stream, slice(2));
    for (auto byte : stream) parser.feed(&byte, 1);
    parser.feed(nullptr, 0, true);
    auto frame = slice(2);
    assert(parser.take(frame.data(), frame.size()) == 6); // Decoder skipped picture 1.
    parser.reset();
    parser.feed(frame.data(), frame.size(), true);
    assert(parser.take(frame.data(), frame.size()) == 0); // No stale SPS after reset.
    auto bad = sps(2); bad.resize(bad.size() - 3);
    stream.clear(); append(stream, bad); append(stream, pps()); append(stream, frame);
    parser.feed(stream.data(), stream.size(), true);
    assert(parser.take(frame.data(), frame.size()) == 0);
    // Parameter-set replacement, not merely a fresh decoder, must clear stale VUI.
    parser.reset();
    auto good = sps(4);
    parser.feed(good.data(), good.size(), true);
    auto parameters = pps(); parser.feed(parameters.data(), parameters.size(), true);
    parser.feed(bad.data(), bad.size(), true);
    parser.feed(frame.data(), frame.size(), true);
    assert(parser.take(frame.data(), frame.size()) == 0);
    // Deterministic malformed NALs exercise the readers under ASan/UBSan.
    uint32_t random = 1;
    for (unsigned i = 0; i < 1000; ++i) {
        parser.reset();
        std::vector<uint8_t> garbage = {0, 0, 1, static_cast<uint8_t>((i % 3 == 0 ? 1 : 33 + i % 2) << 1), 1};
        for (unsigned j = 0; j < i % 256; ++j) {
            random = random * 1664525 + 1013904223;
            garbage.push_back(static_cast<uint8_t>(random >> 24));
        }
        for (auto byte : garbage) parser.feed(&byte, 1);
        parser.feed(nullptr, 0, true);
    }
    parser.reset();
}

int main(int argc, char** argv) {
    unitTests();
    // Optional real Annex B corpus: print one metadata value per coded picture.
    if (argc == 2) {
        std::ifstream file(argv[1], std::ios::binary);
        std::vector<uint8_t> stream{std::istreambuf_iterator<char>(file), {}};
        HevcChroma parser;
        for (size_t i = 0; i < stream.size(); i += 733)
            parser.feed(stream.data() + i, std::min(size_t{733}, stream.size() - i));
        parser.feed(nullptr, 0, true);
        for (size_t i = 0; i + 5 < stream.size(); ++i) {
            if (stream[i] || stream[i + 1] || stream[i + 2] != 1) continue;
            const auto type = (stream[i + 3] >> 1) & 63;
            if (type <= 31 && (stream[i + 5] & 128))
                std::cout << unsigned(parser.take(stream.data() + i, stream.size() - i)) << '\n';
        }
    }
}
