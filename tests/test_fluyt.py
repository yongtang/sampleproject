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
"""test_fluyt"""

import pytest

import functools
import numpy as np
import tensorflow as tf

import fluyt


@fluyt.Transform
def linear(input, *, w, b):
    return tf.matmul(input, w) + b


@pytest.mark.parametrize(
    ("f", "build"),
    [
        pytest.param(
            lambda: linear(
                w=tf.Variable(
                    initial_value=tf.random_normal_initializer()(
                        shape=(1, 1), dtype="float32"
                    ),
                    trainable=True,
                ),
                b=tf.Variable(
                    initial_value=tf.zeros_initializer()(shape=(1,), dtype="float32"),
                    trainable=True,
                ),
            ),
            True,
        ),
        pytest.param(
            lambda: linear(
                w=tf.Variable(
                    initial_value=tf.random_normal_initializer()(
                        shape=(1, 1), dtype="float32"
                    ),
                    trainable=True,
                ),
                b=tf.Variable(
                    initial_value=tf.zeros_initializer()(shape=(1,), dtype="float32"),
                    trainable=True,
                ),
            ),
            False,
        ),
        pytest.param(
            lambda: linear(
                w=fluyt.Param(
                    tf.random_normal_initializer()(shape=(1, 1), dtype="float32"),
                    shape=(1, 1),
                    dtype=tf.float32,
                ),
                b=fluyt.Param(
                    tf.zeros_initializer()(shape=(1,), dtype="float32"),
                    shape=(1,),
                    dtype=tf.float32,
                ),
            ),
            True,
        ),
        pytest.param(
            lambda: linear(
                w=fluyt.Param(
                    tf.random_normal_initializer()(shape=(1, 1), dtype="float32"),
                    shape=(1, 1),
                    dtype=tf.float32,
                ),
                b=fluyt.Param(
                    tf.zeros_initializer()(shape=(1,), dtype="float32"),
                    shape=(1,),
                    dtype=tf.float32,
                ),
            ),
            False,
        ),
    ],
    ids=["variable|build", "variable", "param|build", "param"],
)
def test_layer(f, build):

    np.random.seed(1000)
    tf.random.set_seed(1000)

    y = np.linspace(1, 2, 100).reshape(-1, 1)
    x = y * (1 + np.random.random(100).reshape(-1, 1) * 0.01) / 5

    transform = f()

    if build:
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(1,)),
                fluyt.Layer(transform),
                fluyt.Layer(lambda e: (e * 5.0)),
            ]
        )
        model.summary()
    else:
        model = tf.keras.Sequential(
            [fluyt.Layer(transform), fluyt.Layer(lambda e: (e * 5.0)),]
        )

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=0.05),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    model.fit(x, y, epochs=100, batch_size=32)

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.plot(x, model.predict(x), "b", x, y, "k.")
    # plt.draw()
    # plt.show()

    print("W:", transform._args["w"].value())
    print("B:", transform._args["b"].value())
    assert np.allclose(transform._args["w"].value(), 1.0, atol=0.05)
    assert np.allclose(transform._args["b"].value(), 0.0, atol=0.05)


@pytest.mark.parametrize(
    ("autograph"), [pytest.param(True), pytest.param(False),], ids=["autograph", "no"],
)
def test_function(autograph):

    np.random.seed(1000)
    tf.random.set_seed(1000)

    @fluyt.Transform
    def f(input, e):
        return input + e

    @tf.function(autograph=autograph)
    def g(input):
        return f(e=3.0)()()(input)

    assert f.__name__ == "f"
    assert f(e=3).__name__ == "f"
    assert f(e=3)().__name__ == "f"
    assert f(e=3)()().__name__ == "f"

    assert g(3) == 6


def test_mnist():
    np.random.seed(1000)
    tf.random.set_seed(1000)

    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # model = keras.Sequential(
    #     [
    #         keras.Input(shape=input_shape),
    #         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Flatten(),
    #         layers.Dropout(0.5),
    #         layers.Dense(num_classes, activation="softmax"),
    #     ]
    # )
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            fluyt.Layer(
                fluyt.ops.convolution(ksize=(2, 2), strides=(2, 2)), name="MaxPooling2D"
            ),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            fluyt.Layer(
                fluyt.ops.convolution(ksize=(2, 2), strides=(2, 2)), name="MaxPooling2D"
            ),
            fluyt.Layer(fluyt.ops.reshape(shape=(None, 1600)), name="Flatten"),
            fluyt.Layer(
                fluyt.ops.multiply(right=fluyt.random.bernoulli(p=0.5)), name="Dropout"
            ),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 5

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    assert np.allclose(score[0], 0.061, atol=0.002)
    assert np.allclose(score[1], 0.981, atol=0.002)
