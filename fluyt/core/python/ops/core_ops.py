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
"""fluyt"""

import abc

import tensorflow as tf


class Param:
    def __init__(self, value, shape=None, dtype=None):
        if callable(value):

            def f():
                return value(shape, dtype)

            self._initial = f
        else:

            def g():
                return value

            self._initial = g
        self._reference = None
        super().__init__()

    def reference(self):
        if self._reference is None:
            self._reference = tf.Variable(self._initial(), trainable=True)
        return self._reference

    def value(self):
        return self.reference().value()


class _Transform(abc.ABC):
    @abc.abstractmethod
    def args(self):
        raise NotImplementedError("_Transform.args")


class Transform(_Transform):
    def __init__(self, func, **args):
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self._func = func
        self._args = args

    def __call__(self, *input, **args):
        if len(input) == 0:
            return Transform(self._func, **{**self.args(), **args})
        args = {
            k: (v.value() if isinstance(v, Param) else v)
            for (k, v) in ({**self.args(), **args}).items()
        }
        return self._func(*input, **args)

    def args(self):
        return self._args


class MetaTransform(_Transform):
    def __init__(self, *transform):
        self._transform = transform

    def __call__(self, input):
        value = input
        for f in self._transform:
            value = f(value)
        return value

    def args(self):
        entries = {}
        for f in self._transform:
            for (k, e) in f.args().items():
                entries[f"{f.__name__}_{k}"] = e
        return entries


class _Layer(tf.keras.layers.Layer):
    def __init__(self, *transform):
        transform = tuple(
            (f if isinstance(f, _Transform) else Transform(f)) for f in transform
        )
        if len(transform) == 1:
            self._transform = transform[0]
        else:
            self._transform = MetaTransform(*transform)
        super().__init__()

    def build(self, shape):
        for (k, e) in self._transform.args().items():
            setattr(self, f"_param_{k}", (e.reference() if isinstance(e, Param) else e))
        super().build(shape)

    def call(self, input):
        return self._transform(input)


def Layer(*f, name=None):
    if name is None:
        if len(f) == 1:
            name = type(f[0]).__name__
            name = f[0].__name__ if hasattr(f[0], "__name__") else name
            name = "lambda" if name == "<lambda>" else name
        else:
            name = "MetaTransform"
    return type(name, tuple([_Layer]), dict())(*f)
