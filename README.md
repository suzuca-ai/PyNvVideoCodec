# **PyNvVideoCodec**

[PyNvVideoCodec](https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html) is NVIDIA's Python-based library that provides simple yet powerful Python APIs for hardware-accelerated video encoding and decoding on NVIDIA GPUs. 

PyNvVideoCodec is built on top of the Video Codec SDK and offers encode, decode, and transcode performance on par with it.

The library is distributed under MIT license and is officially supported by NVIDIA. 

This release introduces several new features and enhancements designed to optimize video processing workflows in AI and multimedia applications.

## **Features**

Current PyNvVideoCodec version supports following features:

### **Decode Features**

* **Seek and frame sampling:** Provides efficient and flexible methods for fetching video frames in various modes, including sequential, random, periodic, indexed, batched, and sliced, as well as at a specified target frame rate.  
* **Decoder caching:** Optimizes decoding of short video clips through decoder caching and reconfiguration.  
* **Threaded decoder:** Supports decoding on separate threads, delivering pre-decoded frames with near-zero latency, enabling high-performance video processing pipelines.  
* **Video processing from buffer:** Supports video processing from memory buffers, reducing I/O overhead, enabling streaming applications.  
* **Low latency decode:** Offers zero-latency decoding for video sequences that do not contain B-frames.  
* **SEI extraction:** Supports the extraction of Supplemental Enhancement Information (SEI) messages, allowing access to additional information such as HDR information, timecodes, and custom user data.  
* **Stream metadata access:** Enables access to stream metadata, including frame width, height, bit depth, and keyframe indices, to enhance content management.  
* **GIL handling:** Improved multithreaded performance through better handling of Global Interpreter Lock (GIL) in C++ layer.  
* **Multi-GPU decode:** Enables multi-GPU decoding to efficiently handle larger workloads.  
* **Extended codec support:** Supports codecs H.264, HEVC, AV1, VP8, VP9, VC1, MPEG4, MPEG2, and MPEG1  
* **4:2:2 decode:** Supports 4:2:2 decoding for both H.264 and HEVC formats on Blackwell GPUs (NV16, P210 and P216 surface formats).   
* **Extended output formats :** Decode to various output formats including NV12, YUV420, YUV444, NV16, P010, P016 and RGB24(interleaved and planar)

 

### **Decoded frame color metadata**

Every `DecodedFrame` exposes two read-only properties describing its input bitstream:

* `matrix_coefficients`: the raw CICP matrix code from NVDEC (1 = BT.709,
  2 = unspecified, 5/6 = BT.601). Other codes, including 0, are preserved.
* `video_full_range_flag`: `True` for full range; `False` for limited or
  unspecified range, as reported by NVDEC.

These values belong to the displayed frame, including delayed frames and streams
whose color description changes without changing resolution. They do not describe
any RGB conversion requested with `outputcolortype`. Applications choose their
own policy for unspecified or unsupported matrix codes.

Known NVDEC limitation: with NVIDIA driver 615.71.09, AV1 full-range bitstreams
were reported as limited range by the parser, including with a fresh decoder.
This API preserves the parser result; H.264 and HEVC full-range signaling is
covered by the regression below.

The NVDEC integration regression can be run with
`python -m unittest discover -s tests -p 'test_decoded_frame_color.py' -v`.
It requires an NVIDIA GPU with H.264/HEVC decode support and FFmpeg with libx264/libx265.

### **Encode Features**

* **Encoder reconfiguration:** Supports encoder reconfiguration, enabling dynamic updating of encoding parameters without recreating encoder instances.  
* **SEI insertion:** Allows insertion of SEI messages during encoding.  
* **GIL handling:** Improved multithreaded performance through better handling of Global Interpreter Lock (GIL) in C++ layer.  
* **Multi-GPU encode:** Enables multi-GPU encoding to efficiently handle larger workloads.  
* **Codec support:** Support encoding to codec H.264, HEVC, and AV1.  
* **4:2:2 encode:** Supports 4:2:2 encoding for both H.264 and HEVC formats on Blackwell GPUs (NV16 and P210 surface formats).  
* **Extended input formats:** Encode from various input formats including NV12, YV12, IYUV, YUV444, YUV420\_10BIT, YUV444\_10BIT, NV16, P210, ARGB, ABGR, ARGB10, and ABGR10.

 

### **Transcode Features**

* **Segment-based transcode:** Enables transcoding of video segments based on timestamp ranges, ideal for content editing and partial processing.

## **Distribution**

PyNvVideoCodec library is distributed in two formats: binary distribution via [PyPI](https://pypi.org/project/pynvvideocodec/)  and source code distribution via [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/pynvvideocodec). In both cases, the library and its dependencies can be installed using a single pip install command.

This package on PyPI contains Python WHLs of the PyNvVideoCodec library and sample applications that demonstrate the use of the PyNvVideoCodec API. To install these please open the shell prompt, and run the following command.

```py
$ pip install PyNvVideoCodec
```

## **Sample Applications and Documents**

* A package containing PyNvVideoCodec source code, Python sample applications and documentation can be downloaded from [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/pynvvideocodec).   
* For your convenience, the documents are also accessible online at [PyNvVideoCodec Online Documentation](https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html).