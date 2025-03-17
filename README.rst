PyNvVideoCodec
=============

`PyNvVideoCodec <https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html>`_ is NVIDIA's Python based video codec library for hardware accelerated video encode and decode on NVIDIA GPUs. PyNvVideoCodec is the successor of `VPF <https://github.com/NVIDIA/VideoProcessingFramework>`_ (Video Processing Framework).

The library is distributed under MIT license and is officially supported by NVIDIA. PyNvVideoCodec library internally uses Video Codec SDK's core C/C++ video encode and decode APIs and provides easy to use Python APIs on it. The library offers encode and decode performance close to Video Codec SDK.

Features
--------
Current PyNvVideoCodec version supports following features:

* Codecs: H.264, HEVC, AV1.
* Surface formats: NV12 (8 bit), YUV 4:2:0 (10 bit), YUV 4:4:4 (8 and 10 bit)
* Video container formats: MP4, AVI and MKV
* DLPack support to facilitate data exchange with popular DL frameworks like PyTorch and TensorRT.
* CUDA Array Interface support to facilitate data exchange with NVIDIA's CV-CUDA library.
* CUDA stream support for optimizing throughput.
* Contains Python sample applications demonstrating API usage.

Distribution
-----------
PyNvVideoCodec library is distributed in two formats: binary distribution via `PyPI <https://pypi.org/project/pynvvideocodec/>`_ and source code distribution via `NVIDIA NGC <https://catalog.ngc.nvidia.com/orgs/nvidia/resources/pynvvideocodec>`_. In both cases, the library and its dependencies can be installed using a single pip install command.

This package on PyPI contains Python WHLs of PyNvVideoCodec library. To install this library please open the shell prompt, and run the following command:

.. code-block:: bash

    $ pip install PyNvVideoCodec

Sample Applications and Documents
-------------------------------

* A package containing sample application that demonstrate PyNvVideoCodec API and documents can be downloaded from `NVIDIA NGC <https://catalog.ngc.nvidia.com/orgs/nvidia/resources/pynvvideocodec>`_
* For your convenience, the documents are also accessible online at `PyNvVideoCodec Online Documentation <https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html>`_
