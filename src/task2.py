import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from tensorflow import keras
from tensorflow.keras.activations import softmax

from mnist1d_utils import make_dataset

EPSILON_SINGLE = np.finfo(np.float32).eps


# NOTE(brendan): pytorch functions
class MaskGenerator:
    def __init__(self, shape, sigma, clamp=True):
        self.shape = shape
        self.sigma = sigma
        self.coldness = 20
        self.clamp = clamp

        self.kernel = lambda z: torch.exp(-2 * ((z - 0.5).clamp(min=0) ** 2))

        self.margin = self.sigma
        self.padding = 1 + math.ceil(self.margin + sigma)
        self.radius = 1 + math.ceil(sigma)
        self.shape_in = [math.ceil(z) for z in self.shape]
        self.shape_mid = [
            z + 2 * self.padding - (2 * self.radius + 1) + 1 for z in self.shape_in
        ]
        self.shape_out = [z for z in self.shape_mid]

        self.weight = torch.zeros((1, 2 * self.radius + 1, self.shape_out[0]))

        for k in range(2 * self.radius + 1):
            (u,) = torch.meshgrid(
                torch.arange(self.shape_out[0], dtype=torch.float32),
            )
            i = torch.floor(u) + k - self.padding

            delta = torch.sqrt((u - (self.margin + i)) ** 2)

            self.weight[0, k] = self.kernel(delta / sigma)

    def generate(self, mask_in):
        mask = F.pad(mask_in, pad=2 * (self.padding,))
        mask = mask.unfold(dimension=2, size=2 * self.radius + 1, step=1)
        print(f"mask unfolded torch {mask.shape}")
        mask = mask.reshape(len(mask_in), -1, self.shape_mid[0])
        mask = self.weight * mask

        mask = (mask * F.softmax(self.coldness * mask, dim=1)).sum(dim=1, keepdim=True)

        m = round(self.margin)
        if self.clamp:
            mask = mask.clamp(min=0, max=1)
        cropped = mask[:, :, m : m + self.shape[0]]
        return cropped, mask

    def to(self, dev):
        self.weight = self.weight.to(dev)
        return self


class MaskGenerator_TF:
    def __init__(self, shape, sigma, clamp=True):
        self.shape = shape
        self.sigma = sigma
        self.coldness = 20
        self.clamp = clamp

        self.kernel = lambda z: tf.exp(
            -2 * (tf.clip_by_value(z - 0.5, clip_value_min=0, clip_value_max=1) ** 2)
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

    def generate(self, mask_in):
        paddings = [[0, 0], [0, 0], [self.padding, self.padding]]
        mask = tf.pad(mask_in, paddings)
        # NOTE(brendan): manual unfold for convolution
        mask_unfolded = []
        for i in range(self.shape_mid[0]):
            mask_unfolded.append(mask[:, :, i : i + (2 * self.radius + 1)])
        mask = tf.stack(mask_unfolded, axis=2)
        print(f"mask unfolded tf {mask.shape}")
        # mask = mask.unfold(dimension=2, size=2 * self.radius + 1, step=1)
        mask = tf.reshape(mask, shape=(len(mask_in), -1, self.shape_mid[0]))
        mask = self.weight * mask

        mask = mask * softmax(self.coldness * mask, axis=1)
        mask = tf.reduce_sum(mask, axis=1, keepdims=True)

        m = round(self.margin)
        if self.clamp:
            mask = tf.clip_by_value(mask, clip_value_min=0, clip_value_max=1)
        cropped = mask[:, :, m : m + self.shape[0]]
        return cropped, mask


def imsmooth(
    tensor, sigma, stride=1, padding=0, padding_mode="constant", padding_value=0
):
    assert sigma >= 0
    width = math.ceil(4 * sigma)
    filt = torch.arange(
        -width, width + 1, dtype=torch.float32, device=tensor.device
    ) / (math.sqrt(2) * sigma + EPSILON_SINGLE)
    filt = torch.exp(-filt * filt)
    filt /= torch.sum(filt)
    num_channels = tensor.shape[1]
    width = width + padding
    other_padding = width

    print(f"(torch) tensor {tensor.shape}")
    print(f"(torch) filt {filt.shape}")
    x = F.conv1d(
        tensor,
        filt.reshape((1, 1, -1)).expand(num_channels, -1, -1),
        padding=other_padding,
        stride=stride,
        groups=num_channels,
    )
    print(f"(torch) x {x.shape}")
    return x


class Perturbation:
    def __init__(self, input, num_levels=8, max_blur=20):
        self.num_levels = num_levels
        self.pyramid = []
        assert num_levels >= 2
        assert max_blur > 0
        with torch.no_grad():
            for sigma in torch.linspace(0, 1, self.num_levels):
                y = imsmooth(input, sigma=(1 - sigma) * max_blur)
                self.pyramid.append(y)
            self.pyramid = torch.cat(self.pyramid, dim=0)

    def apply(self, mask):
        n = mask.shape[0]
        w = mask.reshape(n, 1, *mask.shape[1:])
        w = w * (self.num_levels - 1)
        k = w.floor()
        w = w - k
        k = k.long()

        y = self.pyramid[None, :]
        y = y.expand(n, *y.shape[1:])
        k = k.expand(n, 1, *y.shape[2:])
        y0 = torch.gather(y, dim=1, index=k)
        y1 = torch.gather(y, dim=1, index=torch.clamp(k + 1, max=self.num_levels - 1))
        print(f"(torch) y0 {y0.shape} y1 {y1.shape}")

        return ((1 - w) * y0 + w * y1).squeeze(dim=1)

    def to(self, dev):
        self.pyramid.to(dev)
        return self


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

    print(f"(tf) tensor {tensor.shape}")
    print(f"(tf) filt {filt.shape}")
    x = tf.nn.conv1d(
        tf.reshape(tensor, (1, -1, 1)),
        tf.reshape(filt, (-1, 1, 1)),
        stride=stride,
        padding="SAME",
        data_format="NWC",
    )
    print(f"(tf) x {x.shape}")
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
        y = tf.broadcast_to(y, shape=(n, *y.shape[1:]))
        k = tf.broadcast_to(k, shape=(n, 1, *y.shape[2:]))

        y0 = tf.gather_nd(y, indices=k, axis=1)
        y1 = tf.gather_nd(
            y,
            indices=tf.clip_by_value(
                k + 1, clip_value_min=0, clip_value_max=self.num_levels - 1
            ),
            axis=1,
        )
        print(f"(tf) y0 {y0.shape} y1 {y1.shape}")

        return ((1 - w) * y0 + w * y1).squeeze(dim=1)

    def to(self, dev):
        self.pyramid.to(dev)
        return self


def task2():
    model_path = os.path.join("project_a_supp", "models", "MNIST1D.h5")
    model = keras.models.load_model(model_path)
    optimizer = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)

    mnist1d = make_dataset()

    x_test = np.expand_dims(mnist1d["x_test"], axis=-1)
    y_test = mnist1d["y_test"]

    # NOTE(brendan): Extremal perturbation
    areas = [0.1]
    regul_weight = 300
    max_iter = 800
    w = x_test.shape[1]
    # NOTE(brendan): mask step and smoothing
    sigma = 4
    coldness = 20

    # NOTE(brendan): pytorch logic
    pmask = torch.ones(len(areas), 1, w)
    mask_generator = MaskGenerator((w,), sigma)
    pmask.requires_grad_(True)
    mask_, mask = mask_generator.generate(pmask)
    mask_[:] = 0.01

    digit_input = torch.from_numpy(x_test[:1]).float()
    digit_input = digit_input.view((1, 1, w))
    perturbation = Perturbation(digit_input, num_levels=2)
    x = perturbation.apply(mask_)
    plt.subplot(2, 5, 1)
    plt.plot(np.squeeze(digit_input.numpy()), "r")
    plt.axis("off")
    plt.title("Pre (torch)")
    plt.subplot(2, 5, 2)
    plt.plot(np.squeeze(x.detach().numpy()), "r")
    plt.axis("off")
    plt.title("Perturbed (torch)")
    # plt.show()

    # NOTE(brendan): tensorflow logic
    digit_input = x_test[:1].reshape((1, 1, w))
    pmask = tf.ones((len(areas), 1, w))
    mask_generator = MaskGenerator_TF((w,), sigma)
    perturbation = Perturbation_TF(digit_input, num_levels=2)
    for iter_t in range(max_iter):
        mask_, mask = mask_generator.generate(pmask)
        x = perturbation.apply(mask_)
        y = model(x)

    pmask = tf.ones(len(areas), 1, w)
    with tf.GradientTape() as tape:
        tape.watch(pmask)
        mask = pmask
        for iter_t in range(max_iter):
            # NOTE(brendan): generate a smooth
            mask = mask * softmax(coldness * mask, axis=1)
            mask = mask.sum(axis=1, keepdim=True)


if __name__ == "__main__":
    task2()
