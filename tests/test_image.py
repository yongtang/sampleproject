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
"""test_image"""

import os

import numpy as np
import magnify as mf

import tensorflow as tf


def test_image():
    """test_image"""

    # Source from:
    # https://ohmy.disney.com/movies/2016/07/20/twelve-principles-animation-disney
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_image", "Mulan.gif"
    )
    expected = tf.image.decode_gif(tf.io.read_file(path))
    expected = tf.concat([expected, expected], axis=0)

    def fn(image):
        image = mf.Image(dtype=np.float32)(image)
        image = mf.Image(dtype=np.uint8)(image)
        return image

    dataset = tf.data.Dataset.from_tensor_slices([path, path])
    dataset = dataset.map(lambda e: tf.image.decode_gif(tf.io.read_file(e)))

    dataset = dataset.map(fn)

    dataset = dataset.unbatch()
    dataset = dataset.batch(65536)
    image = tf.data.experimental.get_single_element(dataset)

    assert np.array_equal(expected, image)

    # import matplotlib.pyplot as plt
    # for e in tf.unstack(image, axis=0):
    #    plt.imshow(e)
    #    plt.show()
