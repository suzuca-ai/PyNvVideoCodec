"""NVDEC integration regression; requires FFmpeg/libx264/libx265 and a GPU."""

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


class DecodedFrameColorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg"):
            raise unittest.SkipTest("FFmpeg is required")
        try:
            import PyNvVideoCodec as nvc
        except ImportError as exc:
            raise unittest.SkipTest(f"PyNvVideoCodec is unavailable: {exc}")
        cls.nvc = nvc
        cls.sequences = {}
        # Same size throughout: the sequence callback must refresh VUI even
        # when the hardware decoder does not need reconfiguration. 0 and 9
        # also verify that unsupported CICP codes are never silently remapped.
        cls.colors = [(1, False), (6, True), (2, False), (9, True), (5, False), (0, False)]
        cls.frames_per_sequence = 8
        for codec, encoder in (("h264", "libx264"), ("hevc", "libx265")):
            segments = []
            for matrix, full_range in cls.colors:
                command = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=384x256:rate=24",
                    "-frames:v", str(cls.frames_per_sequence), "-an",
                    "-c:v", encoder, "-preset", "fast", "-threads", "1",
                    "-bf", "2", "-g", "24", "-pix_fmt", "yuv420p",
                ]
                if codec == "hevc":
                    command += ["-x265-params", "pools=1:frame-threads=1:log-level=error"]
                # Omit the VUI options for the unspecified case.
                if matrix != 2:
                    command += ["-colorspace", str(matrix), "-color_range", "pc" if full_range else "tv"]
                command += ["-f", codec, "pipe:1"]
                segments.append(subprocess.run(command, check=True, capture_output=True).stdout)
            cls.sequences[codec] = segments

    def check_decode(self, codec, packet_mode, get_frame=False):
        nvc = self.nvc
        decoder = nvc.CreateDecoder(
            gpuid=0, codec=nvc.cudaVideoCodec.H264 if codec == "h264" else nvc.cudaVideoCodec.HEVC,
            cudacontext=0, cudastream=0, usedevicememory=True,
        )
        segments = self.sequences[codec]
        bitstream = b"".join(segments)
        if packet_mode == "single":
            chunks = [bitstream]
        elif packet_mode == "sequence":
            chunks = segments
        else:
            # Deliberately split access units and sequence headers; the native
            # parser, rather than the packet caller, supplies color metadata.
            chunks = [bitstream[i:i + 733] for i in range(0, len(bitstream), 733)]
        frames = []
        snapshots = []
        counts = []
        for chunk in [*chunks, b""]:
            packet = nvc.PacketData()
            storage = ctypes.create_string_buffer(chunk) if chunk else None
            packet.bsl_data = ctypes.addressof(storage) if storage is not None else 0
            packet.bsl = len(chunk)
            if get_frame:
                count = decoder.GetNumDecodedFrame(packet)
                decoded = [decoder.GetFrame() for _ in range(count)]
            else:
                decoded = decoder.Decode(packet)
            counts.append(len(decoded))
            for frame in decoded:
                self.assertIsInstance(frame.matrix_coefficients, int)
                self.assertIsInstance(frame.video_full_range_flag, bool)
                snapshots.append((frame.matrix_coefficients, frame.video_full_range_flag))
            frames.extend(decoded)
        expected = [color for color in self.colors for _ in range(self.frames_per_sequence)]
        self.assertEqual(snapshots, expected)
        # Returned objects retain their metadata when later Decode calls reuse
        # the underlying pixel buffers or encounter a new sequence.
        self.assertEqual([(f.matrix_coefficients, f.video_full_range_flag) for f in frames], expected)
        self.assertGreater(max(counts), 1)
        self.assertGreater(counts[-1], 0, "the regression must exercise delayed output at EOS")
        for name in ("matrix_coefficients", "video_full_range_flag"):
            with self.assertRaises(AttributeError):
                setattr(frames[0], name, 0)

    def test_decode(self):
        for codec in self.sequences:
            for packet_mode in ("single", "sequence", "fragmented"):
                with self.subTest(codec=codec, packet_mode=packet_mode):
                    self.check_decode(codec, packet_mode)

    def test_get_frame(self):
        for codec in self.sequences:
            with self.subTest(codec=codec):
                self.check_decode(codec, "sequence", get_frame=True)

    def test_simple_and_threaded_decoders(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "colors.hevc"
            video = Path(directory) / "colors.mp4"
            raw.write_bytes(self.sequences["hevc"][1])
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-fflags", "+genpts",
                "-i", str(raw), "-c", "copy", str(video),
            ], check=True, capture_output=True)
            expected = [self.colors[1]] * self.frames_per_sequence
            for threaded in (False, True):
                with self.subTest(threaded=threaded):
                    decoder = (self.nvc.ThreadedDecoder(str(video), buffer_size=4) if threaded
                               else self.nvc.SimpleDecoder(str(video)))
                    actual = []
                    try:
                        while len(actual) < len(expected):
                            batch = decoder.get_batch_frames(min(3, len(expected) - len(actual)))
                            self.assertTrue(batch)
                            actual.extend((f.matrix_coefficients, f.video_full_range_flag) for f in batch)
                    finally:
                        if threaded:
                            decoder.end()
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
