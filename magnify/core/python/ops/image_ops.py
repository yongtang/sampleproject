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
"""magnify"""

import tensorflow as tf


class Image:
    def __init__(self, dtype):
        self._dtype = tf.as_dtype(dtype)
        if self._dtype == tf.uint8:

            @tf.function
            def _f(input):
                if input.dtype == tf.uint8:
                    return input
                return tf.cast(input * 255.0, tf.uint8)

            self._function = _f
        else:

            @tf.function
            def _f(input):
                if input.dtype == tf.uint8:
                    return tf.cast(input, self._dtype) / 255.0
                return tf.cast(input, self._dtype)

            self._function = _f

    def __call__(self, input):
        input = tf.convert_to_tensor(input)
        return self._function(input)
