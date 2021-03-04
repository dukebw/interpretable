import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import softmax

from mnist1d_utils import make_dataset

EPSILON_SINGLE = np.finfo(np.float32).eps


class MaskGenerator_TF:
    def __init__(self, shape, sigma, clamp=True):
        self.shape = shape
        self.sigma = sigma
        self.coldness = 20
        self.clamp = clamp

        self.kernel = lambda z: tf.exp(
            -2
            * (tf.clip_by_value(z - 0.5, clip_value_min=0, clip_value_max=np.inf) ** 2)
        )

        self.margin = self.sigma
        self.padding = 1 + math.ceil(self.margin + sigma)
        self.radius = 1 + math.ceil(sigma)
        self.shape_in = [math.ceil(z) for z in self.shape]
        self.shape_mid = [
            z + 2 * self.padding - (2 * self.radius + 1) + 1 for z in self.shape_in
        ]
        self.shape_out = [z for z in self.shape_mid]

        self.weight = np.zeros((1, 2 * self.radius + 1, self.shape_out[0]))

        for k in range(2 * self.radius + 1):
            (u,) = np.meshgrid(np.arange(self.shape_out[0]))
            i = np.floor(u) + k - self.padding

            delta = np.sqrt((u - (self.margin + i)) ** 2)

            self.weight[0, k] = self.kernel(delta / sigma)
        self.weight = tf.convert_to_tensor(self.weight, dtype=tf.float32)
        print(f"(tf) weight {self.weight.shape}")
        print(self.weight)

    def generate(self, mask_in):
        paddings = [[0, 0], [0, 0], [self.padding, self.padding]]
        mask = tf.pad(mask_in, paddings)
        # NOTE(brendan): this unfolding is for convolution
        mask_unfolded = []
        for i in range(self.shape_mid[0]):
            mask_unfolded.append(mask[:, :, i : i + (2 * self.radius + 1)])
        mask = tf.stack(mask_unfolded, axis=3)
        mask = tf.squeeze(mask, axis=1)
        # NOTE(brendan): convolve Gaussian weights with mask (smoothness inductive bias)
        mask = self.weight * mask

        mask = mask * softmax(self.coldness * mask, axis=1)
        mask = tf.reduce_sum(mask, axis=1, keepdims=True)

        m = round(self.margin)
        if self.clamp:
            mask = tf.clip_by_value(mask, clip_value_min=0, clip_value_max=1)
        cropped = mask[:, :, m : m + self.shape[0]]
        return cropped, mask


def imsmooth_TF(
    tensor, sigma, stride=1, padding=0, padding_mode="constant", padding_value=0
):
    assert sigma >= 0
    width = math.ceil(4 * sigma)
    filt = np.arange(-width, width + 1) / (math.sqrt(2) * sigma + EPSILON_SINGLE)
    filt = tf.exp(-filt * filt)
    filt /= tf.reduce_sum(filt)
    num_channels = tensor.shape[1]
    assert num_channels == 1
    width = width + padding
    other_padding = width

    x = tf.nn.conv1d(
        tf.reshape(tensor, (1, -1, 1)),
        tf.reshape(filt, (-1, 1, 1)),
        stride=stride,
        padding="SAME",
        data_format="NWC",
    )
    return tf.reshape(x, (1, 1, -1))


class Perturbation_TF:
    def __init__(self, input, num_levels=8, max_blur=20):
        self.num_levels = num_levels
        self.pyramid = []
        assert num_levels >= 2
        assert max_blur > 0
        for sigma in np.linspace(0, 1, self.num_levels):
            y = imsmooth_TF(input, sigma=(1 - sigma) * max_blur)
            self.pyramid.append(y)
        self.pyramid = tf.concat(self.pyramid, axis=0)

    def apply(self, mask):
        n = mask.shape[0]
        w = tf.reshape(mask, (n, 1, *mask.shape[1:]))
        w = w * (self.num_levels - 1)
        k = tf.floor(w)
        w = w - k
        k = tf.cast(k, dtype=tf.int64)

        y = self.pyramid[None, :]
        y = tf.cast(y, tf.float32)
        y = tf.broadcast_to(y, shape=(n, *y.shape[1:]))
        k = tf.broadcast_to(k, shape=(n, 1, *y.shape[2:]))

        first_idx = tf.zeros(k.shape, dtype=tf.int64)
        third_idx = tf.zeros(k.shape, dtype=tf.int64)
        fourth_idx = tf.range(y.shape[-1], dtype=tf.int64)
        fourth_idx = tf.reshape(fourth_idx, k.shape)
        indices = tf.stack((first_idx, k, third_idx, fourth_idx), axis=-1)
        y0 = tf.gather_nd(y, indices=indices)

        k = tf.clip_by_value(
            k + 1, clip_value_min=0, clip_value_max=self.num_levels - 1
        )
        indices = tf.stack((first_idx, k, third_idx, fourth_idx), axis=-1)
        y1 = tf.gather_nd(y, indices=indices)

        return tf.squeeze((1 - w) * y0 + w * y1, axis=1)

    def to(self, dev):
        self.pyramid.to(dev)
        return self


def task2():
    model_path = os.path.join("project_a_supp", "models", "MNIST1D.h5")
    model = keras.models.load_model(model_path)
    # NOTE(brendan): remove softmax to optimize class logit (linear gradient)
    model.layers[-1].activation = None
    optimizer = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)

    mnist1d = make_dataset()

    x_test = np.expand_dims(mnist1d["x_test"], axis=-1)
    y_test = mnist1d["y_test"]

    # NOTE(brendan): Extremal perturbation hyperparameters
    areas = [0.3]
    regul_weight = 300
    max_iter = 800
    w = x_test.shape[1]
    sigma = 4
    num_levels = 2

    digit_input = x_test[:1].reshape((1, 1, w))
    mask_generator = MaskGenerator_TF((w,), sigma)
    perturbation = Perturbation_TF(digit_input, num_levels=num_levels, max_blur=20)

    # NOTE(brendan): Prepare reference area vector
    max_area = np.prod(mask_generator.shape_out)
    reference = np.ones((len(areas), max_area))
    for i, a in enumerate(areas):
        reference[i, : int(max_area * (1 - a))] = 0
    reference = tf.convert_to_tensor(reference, dtype=tf.float32)

    pmask = tf.ones((len(areas), 1, w))
    pmask = tf.Variable(pmask)
    y = model(tf.reshape(digit_input, (1, w, 1)))
    target_channel = np.argmax(tf.squeeze(y))
    for iter_t in range(max_iter):
        with tf.GradientTape() as tape:
            # NOTE(brendan): generate mask from smooth manifold
            mask_, mask = mask_generator.generate(pmask)

            # NOTE(brendan): use "preserve" variant of EP
            x = perturbation.apply(mask_)

            x = tf.reshape(x, (1, w, 1))
            y = model(x)

            reward = y[:, target_channel]

            # NOTE(brendan): Area regularization
            mask_sorted = tf.reshape(mask, (len(areas), -1))
            mask_sorted = tf.sort(mask_sorted, axis=1)
            regul = -((mask_sorted - reference) ** 2)
            regul = regul_weight * tf.reduce_mean(regul, axis=1)
            energy = tf.reduce_sum(reward + regul)

            grads = tape.gradient(-energy, pmask)
        optimizer.apply_gradients(zip([grads], [pmask]))
        pmask = tf.clip_by_value(pmask, clip_value_min=0, clip_value_max=1)
        pmask = tf.Variable(pmask)

        # NOTE(brendan): the area constraint tends towards a hard constraint
        regul_weight *= 1.0035

        if (iter_t % 100) == 0:
            print(f"grads {grads}")
            print(f"regul {regul} reward {reward}")
            print(f"pmask after: {pmask}")
            print(f"mask_sorted {mask_sorted}")
            print(f"reference {reference}")

    plt.subplot(2, 5, 1)
    plt.plot(np.squeeze(digit_input), "r")
    plt.axis("off")
    plt.title("Input")
    plt.subplot(2, 5, 2)
    plt.plot(np.squeeze(x.numpy()), "r")
    plt.axis("off")
    plt.title("Perturbed")
    plt.show()


if __name__ == "__main__":
    task2()
