# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""

"""

__author__ = "NVIDIA"
__copyright__ = "Copyright 2024, NVIDIA"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "NVIDIA"
__email__ = "TODO"
__status__ = "Production"


from ast import Str
import string


try:
    # Import native module
    from ._PyNvVideoCodec import *  # noqa
    from enum import Enum
except ImportError:
    import distutils.sysconfig
    from os.path import join, dirname
    raise RuntimeError("Failed to import native module _PyNvVideoCodec! "
                           f"Please check whether \"{join(dirname(__file__), '_PyNvVideoCodec' + distutils.sysconfig.get_config_var('EXT_SUFFIX'))}\""  # noqa
                           " exists and can find all library dependencies (CUDA, ffmpeg).\n"
                           "On Unix systems, you can use `ldd` on the file to see whether it can find all dependencies.\n"
                           "On Windows, you can use \"dumpbin /dependents\" in a Visual Studio command prompt or\n"
                           "https://github.com/lucasg/Dependencies/releases.")

def tostring(val):
    if type(val) == dict:
        x  = dict(map(lambda x: (x[0], tostring(x[1])), val.items()))
    else:
        x = str(val)
        x = x.replace("]", "")
        x = x.replace("[", "")
    return x

def format_optional_params(optional_param):
    param  = tostring(optional_param)

    if "slice" in param:
        param["slice_mode"] = param["slice"]["mode"]
        param["slice_data"] = param["slice"]["data"]
        del param["slice"]

    if "timinginfo" in param:
        param["num_unit_in_ticks"] = param["timinginfo"]["num_unit_in_ticks"]
        param["timescale"] = param["timinginfo"]["timescale"]
        del param["timinginfo"]

    return param

class Codec(Enum):
    h264 = 4
    hevc = 8
    av1 = 11

def CreateEncoder(
   width ,
   height,
   fmt ,
   usecpuinputbuffer,
   **kwargs
   ):
    cudacontext = 0
    cudastream = 0
    if "cudacontext" in kwargs:
        cudacontext = int(kwargs["cudacontext"])
        del kwargs["cudacontext"]
    if "cudastream" in kwargs:
        cudastream = int(kwargs["cudastream"])
        del kwargs["cudastream"]

    optional_args = format_optional_params(kwargs)

    return PyNvEncoder(width, height , fmt, cudacontext, cudastream,  usecpuinputbuffer ,optional_args)