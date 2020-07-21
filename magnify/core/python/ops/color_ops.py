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
"""magnify.color"""

import tensorflow as tf

from magnify.core.python.ops.core_ops import _FunctionalImplementation, Input


class RGB2Grayscale(_FunctionalImplementation):
    def __init__(self):
        super().__init__()

    def _annotate(self, *input):
        # unpack input
        (input,) = input
        dtype = input.dtype
        return (Input(dtype=dtype),)

    def _function(self, *input):
        # unpack input tuple
        (input,) = input
        value = tf.image.convert_image_dtype(input, tf.float32)
        coeff = [0.2125, 0.7154, 0.0721]
        value = tf.tensordot(value, coeff, (-1, -1))
        value = tf.expand_dims(value, -1)
        output = tf.image.convert_image_dtype(value, input.dtype)
        return (output,)
