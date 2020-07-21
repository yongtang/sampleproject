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

import abc

import tensorflow as tf


class Input:
    def __init__(self, dtype):
        self._dtype = tf.dtypes.as_dtype(dtype)

    @property
    def dtype(self):
        return self._dtype


class _FunctionalPrototype(abc.ABC):
    def apply(self, functional):
        return self._apply(functional)

    @abc.abstractmethod
    def __call__(self, *input):
        pass

    @abc.abstractmethod
    def annotate(self, *input):
        pass

    @abc.abstractmethod
    def _function(self, *input):
        pass

    @abc.abstractmethod
    def _apply(self, functional):
        pass


class Functional(_FunctionalPrototype):
    def __init__(self):
        self._entries = []
        super().__init__()

    def __call__(self, *input):
        if not hasattr(self, "_input"):
            input = tuple([tf.convert_to_tensor(e) for e in input])
            return (
                Functional()
                .apply(self)
                .annotate(*tuple([Input(e.dtype) for e in input]))
                .__call__(*input)
            )

        input = tuple(
            [tf.convert_to_tensor(e, dtype=i.dtype) for e, i in zip(input, self._input)]
        )

        output = self._function(*input)
        return output[0] if len(output) == 1 else output

    def annotate(self, *input):
        output = input
        for e in self._entries:
            output = e._annotate(*output)
        self._input, self._output = input, output
        return self

    def _function(self, *input):
        output = input
        for e in self._entries:
            output = e._function(*output)
        return output

    def _apply(self, functional):
        if isinstance(functional, Functional):
            self._entries.extend(functional._entries)
        else:
            self._entries.append(functional)
        if hasattr(self, "_input"):
            self.annotate(*self._input)
        return self


class _FunctionalImplementation(_FunctionalPrototype):
    def __init__(self):
        super().__init__()

    def __call__(self, *input):
        return Functional().apply(self).__call__(*input)

    def _apply(self, functional):
        return Functional().apply(self).apply(functional)

    def annotate(self, *input):
        return Functional().apply(self).annotate(*input)

    @abc.abstractmethod
    def _function(self, *input):
        pass


class Image(_FunctionalImplementation):
    def __init__(self, dtype, color=None):
        self._dtype = tf.dtypes.as_dtype(dtype)
        self._color = color
        super().__init__()

    def _annotate(self, *input):
        return (Input(dtype=self._dtype),)

    def _function(self, *input):
        # unpack input tuple
        (input,) = input
        assert input.dtype in (tf.uint8, tf.float16, tf.float32, tf.float64)
        if self._dtype.is_floating and not input.dtype.is_floating:
            return (
                (tf.cast(input, self._dtype) / tf.cast(input.dtype.max, self._dtype)),
            )
        elif input.dtype.is_floating and not self._dtype.is_floating:
            return (
                (tf.cast(input * tf.cast(input.dtype.max, input.dtype), self._dtype)),
            )
        return ((tf.cast(input, self._dtype)),)
