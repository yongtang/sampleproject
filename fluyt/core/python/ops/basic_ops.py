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
"""fluyt.ops"""

import functools
import operator

import tensorflow as tf

from fluyt.core.python.ops.core_ops import Transform


@Transform
def reshape(input, *, shape, name=None):
    shape = [-1 if e is None else e for e in shape]
    return tf.reshape(input, shape)


@Transform
def multiply(input, *, right, name=None):
    right = right(tf.shape(input), input.dtype) if callable(right) else right
    return tf.math.multiply(input, right)
