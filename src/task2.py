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


# NOTE(brendan): from MNIST1D.ipynb
def get_digit_templates():
    d0 = np.asarray([5, 6, 6.5, 6.75, 7, 7, 7, 7, 6.75, 6.5, 6, 5])
    d1 = np.asarray([5, 3, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 5, 5])
    d2 = np.asarray([5, 6, 6.5, 6.5, 6, 5.25, 4.75, 4, 3.5, 3.5, 4, 5])
    d3 = np.asarray([5, 6, 6.5, 6.5, 6, 5, 5, 6, 6.5, 6.5, 6, 5])
    d4 = np.asarray([5, 4.4, 3.8, 3.2, 2.6, 2.6, 5, 5, 5, 5, 5, 5])
    d5 = np.asarray([5, 3, 3, 3, 3, 5, 6, 6.5, 6.5, 6, 4.5, 5])
    d6 = np.asarray([5, 4, 3.5, 3.25, 3, 3, 3, 3, 3.25, 3.5, 4, 5])
    d7 = np.asarray([5, 7, 7, 6.6, 6.2, 5.8, 5.4, 5, 4.6, 4.2, 5, 5])
    d8 = np.asarray([5, 4, 3.5, 3.5, 4, 5, 5, 4, 3.5, 3.5, 4, 5])
    d9 = np.asarray([5, 4, 3.5, 3.5, 4, 5, 5, 5, 5, 4.7, 4.3, 5])

    x = np.stack([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])
    x -= x.mean(1, keepdims=True)  # whiten
    x /= x.std(1, keepdims=True)
    x -= x[:, :1]  # signal starts and ends at 0

    templates = {
        "x": x / 6.0,
        "t": np.linspace(-5, 5, len(d0)) / 6.0,
        "y": np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    }
    return templates


def task2():
    model_path = os.path.join("project_a_supp", "models", "MNIST1D.h5")
    model = keras.models.load_model(model_path)
    # NOTE(brendan): remove softmax to optimize class logit (linear gradient)
    model.layers[-1].activation = None
    optimizer = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)

    mnist1d = make_dataset()

    x_test = np.expand_dims(mnist1d["x_test"], axis=-1)
    y_test = mnist1d["y_test"]

    digit_templates = get_digit_templates()

    # NOTE(brendan): Extremal perturbation hyperparameters
    areas = [0.3]
    initial_regul_weight = 300
    max_iter = 800
    w = x_test.shape[1]
    sigma = 4
    num_levels = 2

    # NOTE(brendan): just to figure out correct/incorrect predictions for visualization
    correct_indices = []
    incorrect_indices = []
    for i in range(len(x_test)):
        digit_input = x_test[i : i + 1]
        digit_label = y_test[i : i + 1]
        digit_prediction = model(digit_input).numpy()

        digit_prediction = np.argmax(digit_prediction)
        if digit_prediction == digit_label:
            correct_indices.append(i)
            continue
        incorrect_indices.append(i)

    num_vis_examples = 10
    visualization_type = "incorrect"
    if visualization_type == "correct":
        example_indices = correct_indices
    else:
        example_indices = incorrect_indices
    for vis_idx, example_idx in enumerate(example_indices[:num_vis_examples]):
        regul_weight = initial_regul_weight
        # NOTE(brendan): Extremal perturbation algorithm
        digit_input = x_test[example_idx : example_idx + 1].reshape((1, 1, w))
        mask_generator = MaskGenerator_TF((w,), sigma)
        perturbation = Perturbation_TF(digit_input, num_levels=num_levels, max_blur=20)

        # NOTE(brendan): Prepare reference area vector
        max_area = np.prod(mask_generator.shape_out)
        reference = np.ones((len(areas), max_area))
        for area_idx, area in enumerate(areas):
            reference[area_idx, : int(max_area * (1 - area))] = 0
        reference = tf.convert_to_tensor(reference, dtype=tf.float32)

        pmask = tf.ones((len(areas), 1, w))
        pmask = tf.Variable(pmask)
        y = model(tf.reshape(digit_input, (1, w, 1)))
        target_channel = np.argmax(tf.squeeze(y))
        for iter_t in range(max_iter):
            with tf.GradientTape() as tape:
                # NOTE(brendan): generate mask from smooth manifold
                mask_, mask = mask_generator.generate(pmask)

                # NOTE(brendan): use "preserve" variant of EP to perturb input
                # with smooth mask
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
                print(f"regul {regul} reward {reward}")
                print(f"pmask after: {pmask}")
                print(f"mask_sorted {mask_sorted}")
                print(f"reference {reference}")

        if visualization_type == "correct":
            num_rows = 3
        else:
            num_rows = 4
        plt.subplot(num_rows, num_vis_examples, 1 + vis_idx)
        plt.plot(np.squeeze(digit_input), "r")
        plt.axis("off")
        plt.title("Input")

        plt.subplot(num_rows, num_vis_examples, 1 + num_vis_examples + vis_idx)
        plt.plot(np.squeeze(x.numpy()), "r")
        plt.axis("off")
        plt.title("Perturbed")

        plt.subplot(num_rows, num_vis_examples, 1 + (2 * num_vis_examples) + vis_idx)
        plt.plot(digit_templates["x"][y_test[example_idx]], "r")
        plt.axis("off")
        plt.title(f"Template (label: {y_test[example_idx]})")

        if visualization_type == "incorrect":
            plt.subplot(
                num_rows, num_vis_examples, 1 + (3 * num_vis_examples) + vis_idx
            )
            plt.plot(digit_templates["x"][target_channel], "r")
            plt.axis("off")
            plt.title(f"Template (predicted: {target_channel})")
    plt.show()


if __name__ == "__main__":
    task2()
