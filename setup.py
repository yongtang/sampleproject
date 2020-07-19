# Copyright 2020 Yong Tang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup for pip package."""

import os
import sys
import shutil
import tempfile
import fnmatch
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

project = "magnify"
version = "0.1.0"


class BinaryDistribution(setuptools.dist.Distribution):
    def has_ext_modules(self):
        return True


setuptools.setup(
    name=project,
    version=version,
    packages=setuptools.find_packages(where=".", exclude=["tests"]),
    python_requires=">=3.6, <3.9",
    install_requires=["tensorflow"],
    zip_safe=False,
    distclass=BinaryDistribution,
)
