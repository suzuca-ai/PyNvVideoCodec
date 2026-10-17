# SPDX-FileCopyrightText: Copyright (c) 2026 Suzuca Technologies
# SPDX-License-Identifier: MIT
"""HEVC per-picture chroma regression; requires FFmpeg/libx265 and NVDEC."""

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


class DecodedFrameChromaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg"):
            raise unittest.SkipTest("FFmpeg is required")
        try:
            import PyNvVideoCodec as nvc
        except ImportError as exc:
            raise unittest.SkipTest(f"PyNvVideoCodec is unavailable: {exc}")
        cls.nvc = nvc
        cls.sequences = []
        # Identical dimensions and color metadata: only chroma location changes.
        # SPS IDs are deliberately reused across sequences, including one packet.
        for location in (None, 0, 1, 2, 3, 4, 5, None):
            options = "pools=1:frame-threads=1:log-level=error"
            if location is not None:
                options += f":chromaloc={location}"
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc2=size=384x256:rate=24", "-frames:v", "8", "-an",
                "-c:v", "libx265", "-preset", "fast", "-threads", "1", "-bf", "2",
                "-g", "24", "-pix_fmt", "yuv420p", "-x265-params", options,
                "-f", "hevc", "pipe:1",
            ]
            cls.sequences.append(subprocess.run(command, check=True, capture_output=True).stdout)
        cls.expected = [location for location in (1, 1, 2, 3, 4, 5, 6, 1) for _ in range(8)]

    def check_decode(self, chunk_size=None, get_frame=False):
        nvc = self.nvc
        decoder = nvc.CreateDecoder(gpuid=0, codec=nvc.cudaVideoCodec.HEVC,
                                   cudacontext=0, cudastream=0, usedevicememory=True)
        stream = b"".join(self.sequences)
        chunks = ([stream] if chunk_size is None else
                  [stream[i:i + chunk_size] for i in range(0, len(stream), chunk_size)])
        frames, snapshots, counts = [], [], []
        for chunk in [*chunks, b""]:
            storage = ctypes.create_string_buffer(chunk) if chunk else None
            packet = nvc.PacketData()
            packet.bsl_data = ctypes.addressof(storage) if storage else 0
            packet.bsl = len(chunk)
            if get_frame:
                decoded = [decoder.GetFrame() for _ in range(decoder.GetNumDecodedFrame(packet))]
            else:
                decoded = decoder.Decode(packet)
            counts.append(len(decoded))
            for frame in decoded:
                self.assertIsInstance(frame.chroma_location, int)
                snapshots.append(frame.chroma_location)
            frames.extend(decoded)
        self.assertEqual(snapshots, self.expected)
        self.assertEqual([f.chroma_location for f in frames], self.expected)
        self.assertGreater(counts[-1], 0, "exercise delayed B-frame output at EOS")
        with self.assertRaises(AttributeError):
            frames[0].chroma_location = 0

    def test_decode(self):
        for chunk_size in (None, 733, 7):
            with self.subTest(chunk_size=chunk_size):
                self.check_decode(chunk_size)

    def test_get_frame(self):
        self.check_decode(733, get_frame=True)

    def test_timestamps(self):
        # Demuxed access units have non-monotonic PTS in decode order (B-frames).
        with tempfile.TemporaryDirectory() as directory:
            raw, video = Path(directory) / "input.hevc", Path(directory) / "input.mp4"
            raw.write_bytes(self.sequences[2])
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-fflags", "+genpts",
                "-i", str(raw), "-c", "copy", str(video),
            ], check=True, capture_output=True)
            for fragmented, end_picture in ((False, False), (True, False), (False, True)):
                with self.subTest(fragmented=fragmented, end_picture=end_picture):
                    decoder = self.nvc.CreateDecoder(gpuid=0, codec=self.nvc.cudaVideoCodec.HEVC,
                        cudacontext=0, cudastream=0, usedevicememory=True)
                    expected, actual = [], []
                    for packet in self.nvc.CreateDemuxer(filename=str(video)):
                        if packet.bsl:
                            expected.append(packet.pts)
                        data = ctypes.string_at(packet.bsl_data, packet.bsl) if packet.bsl else b""
                        chunks = [data[:7], data[7:]] if fragmented and data else [data]
                        for index, chunk in enumerate(chunks):
                            storage = ctypes.create_string_buffer(chunk) if chunk else None
                            item = self.nvc.PacketData()
                            item.bsl_data = ctypes.addressof(storage) if storage else 0
                            item.bsl = len(chunk)
                            item.pts = packet.pts if index == 0 else -9223372036854775808
                            if end_picture and chunk:
                                item.decode_flag = self.nvc.VideoPacketFlag.ENDOFPICTURE
                            actual.extend(frame.timestamp for frame in decoder.Decode(item))
                    actual.extend(frame.timestamp for frame in decoder.Decode(self.nvc.PacketData()))
                    self.assertEqual(actual, sorted(expected))

    def test_discontinuity(self):
        nvc = self.nvc
        decoder = nvc.CreateDecoder(gpuid=0, codec=nvc.cudaVideoCodec.HEVC,
                                   cudacontext=0, cudastream=0, usedevicememory=True)
        for index, expected in ((6, 6), (2, 2)):
            reset = nvc.PacketData()
            reset.bDiscontinuity = True
            self.assertEqual(decoder.GetNumDecodedFrame(reset), 0)
            actual = []
            for chunk in (self.sequences[index], b""):
                storage = ctypes.create_string_buffer(chunk) if chunk else None
                packet = nvc.PacketData()
                packet.bsl_data = ctypes.addressof(storage) if storage else 0
                packet.bsl = len(chunk)
                actual.extend(decoder.GetFrame().chroma_location
                              for _ in range(decoder.GetNumDecodedFrame(packet)))
            self.assertEqual(actual, [expected] * 8)


if __name__ == "__main__":
    unittest.main()
