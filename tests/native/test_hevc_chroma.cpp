// SPDX-FileCopyrightText: Copyright (c) 2026 Suzuca Technologies
// SPDX-License-Identifier: MIT
#include "HevcChroma.h"
#include "HevcAccessUnits.h"
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

static std::vector<uint8_t> sps(int top, int bottom = -2, bool complex = false, bool oversized = false) {
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
    w.ue(oversized ? 1 : complex ? 2 : 0);
    if (oversized) { w.ue(0xfffffffeU); w.ue(0); }
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

static std::vector<uint8_t> pps(unsigned source = 0) { Writer w; w.ue(0); w.ue(source); return w.nal(34); }
static std::vector<uint8_t> slice(unsigned id, bool first = true) {
    Writer w; w.put(first); w.ue(0); w.put(id, 16); return w.nal(1);
}
static void append(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src) { dst.insert(dst.end(), src.begin(), src.end()); }

static void unitTests() {
    HevcChroma parser;
    const auto parse = [&](const std::vector<uint8_t>& bytes) { return parser.parse(bytes.data(), bytes.size()); };
    for (bool complex : {false, true}) {
        for (int value : {-2, -1, 0, 1, 2, 3, 4, 5, 6}) {
            std::vector<uint8_t> unit;
            append(unit, sps(value, -2, complex)); append(unit, pps());
            append(unit, slice(1)); append(unit, slice(999, false));
            assert(parse(unit) == (value < 0 ? 1 : value <= 5 ? value + 1 : 0));
        }
    }
    parse(sps(0, 1)); assert(parse(slice(1)) == 0);
    // Byte-identical slices following a skipped picture cannot inherit its VUI.
    parse(sps(2)); parse(pps()); assert(parse(slice(1)) == 3);
    parse(sps(5)); assert(parse(slice(1)) == 6);
    parser.reset(); assert(parse(slice(1)) == 0);
    auto bad = sps(2); bad.resize(bad.size() - 3);
    parse(sps(4)); parse(pps()); parse(bad); assert(parse(slice(1)) == 0);
    parse(sps(4)); parse(pps()); parse(pps(16)); assert(parse(slice(1)) == 0);
    parse(sps(2, -2, false, true)); parse(pps()); assert(parse(slice(1)) == 0);
    // Deterministic malformed NALs exercise the readers under ASan/UBSan.
    uint32_t random = 1;
    for (unsigned i = 0; i < 1000; ++i) {
        parser.reset();
        std::vector<uint8_t> garbage = {0, 0, 1, static_cast<uint8_t>((i % 3 == 0 ? 1 : 33 + i % 2) << 1), 1};
        for (unsigned j = 0; j < i % 256; ++j) {
            random = random * 1664525 + 1013904223;
            garbage.push_back(static_cast<uint8_t>(random >> 24));
        }
        parse(garbage);
    }
    // No pictures are queued, including a large batch of rejected minimal NALs.
    HevcAccessUnits units;
    std::vector<uint8_t> minimal;
    for (int i = 0; i < 100000; ++i) append(minimal, slice(1));
    size_t seen = 0;
    units.feed(minimal.data(), minimal.size(), 123, true,
        [&](const uint8_t* data, int size, int64_t) { parser.parse(data, size); ++seen; });
    assert(seen == 100000);
    // An unterminated AU is bounded even if no picture can be identified.
    units.reset();
    std::vector<uint8_t> zeros(65536);
    bool rejected = false;
    try {
        for (unsigned i = 0; i < 1025; ++i)
            units.feed(zeros.data(), zeros.size(), 0, false, [](const uint8_t*, int, int64_t) { assert(false); });
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    units.reset();
}

int main(int argc, char** argv) {
    av_log_set_level(AV_LOG_QUIET);
    unitTests();
    // Real Annex B corpus verifies FFmpeg framing at arbitrary packet boundaries.
    if (argc == 2) {
        std::ifstream file(argv[1], std::ios::binary);
        std::vector<uint8_t> stream{std::istreambuf_iterator<char>(file), {}};
        std::vector<uint8_t> expected;
        for (size_t chunk : {size_t{1}, size_t{7}, size_t{733}, stream.size()}) {
            HevcAccessUnits units;
            HevcChroma parser;
            std::vector<uint8_t> actual;
            auto consume = [&](const uint8_t* data, int size, int64_t) { actual.push_back(parser.parse(data, size)); };
            for (size_t i = 0; i < stream.size(); i += chunk)
                units.feed(stream.data() + i, std::min(chunk, stream.size() - i), 0, false, consume);
            units.feed(nullptr, 0, 0, true, consume);
            if (expected.empty()) expected = actual;
            assert(actual == expected);
        }
        std::vector<std::vector<uint8_t>> frames;
        HevcAccessUnits splitter;
        splitter.feed(stream.data(), stream.size(), 0, true,
            [&](const uint8_t* data, int size, int64_t) { frames.emplace_back(data, data + size); });
        for (int mode : {0, 1, 2}) {
            HevcAccessUnits units;
            std::vector<int64_t> times;
            auto consume = [&](const uint8_t*, int, int64_t pts) { times.push_back(pts); };
            for (size_t i = 0; i < frames.size(); ++i) {
                const auto& frame = frames[i];
                const auto prefix = mode == 1 ? std::min(size_t{7}, frame.size()) : frame.size();
                units.feed(frame.data(), prefix, 1000 + i, mode == 2, consume);
                if (prefix < frame.size())
                    units.feed(frame.data() + prefix, frame.size() - prefix, AV_NOPTS_VALUE, false, consume);
            }
            units.feed(nullptr, 0, AV_NOPTS_VALUE, true, consume);
            assert(times.size() == frames.size());
            for (size_t i = 0; i < times.size(); ++i) assert(times[i] == 1000 + static_cast<int64_t>(i));
        }
        for (auto value : expected) std::cout << unsigned(value) << '\n';
    }
}
