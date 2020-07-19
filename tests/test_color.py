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
"""test_color"""

import pytest
import skimage.color

import numpy as np
import magnify as mf

import tensorflow as tf


@pytest.mark.parametrize(
    ("data", "transform", "check"),
    [
        pytest.param(
            lambda: (np.random.random((5, 10, 20, 3)) * 255.0).astype(np.uint8),
            mf.color.Grayscale(),
            lambda e: tf.expand_dims(
                tf.cast(skimage.color.rgb2gray(e) * 255.0, tf.uint8), axis=-1
            ),
        ),
    ],
    ids=["rgb_to_grayscale",],
)
def test_color(data, transform, check):
    """test_color"""

    np.random.seed(1000)

    input = data()
    expected = check(input)

    @tf.function(autograph=False)
    def f(image):
        image = mf.Image(dtype=np.uint8)(image)
        image = transform(image)
        image = mf.Image(dtype=np.uint8)(image)
        return image

    image = f(input)
    assert np.allclose(expected, image, atol=1.0)


@pytest.mark.parametrize(
    ("data", "transform", "check"),
    [
        pytest.param(
            lambda: (np.random.random((5, 10, 20, 3))).astype(np.float32),
            mf.color.Grayscale(),
            lambda e: tf.expand_dims(skimage.color.rgb2gray(e), axis=-1),
        ),
    ],
    ids=["rgb_to_grayscale",],
)
def test_color_float(data, transform, check):
    """test_color_float"""

    np.random.seed(1000)

    for dtype in [np.float16, np.float32, np.float64]:
        input = data().astype(dtype)
        expected = tf.cast(check(input), tf.dtypes.as_dtype(dtype))

        @tf.function(autograph=False)
        def f(image):
            image = mf.Image(dtype=dtype)(image)
            image = transform(image)
            image = mf.Image(dtype=dtype)(image)
            return image

        image = f(input)
        assert np.allclose(expected, image, atol=0.01), "dtype {} failed".format(dtype)
