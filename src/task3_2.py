import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPSILON_SINGLE = np.finfo(np.float32).eps


class MaskGenerator_TF:
    def __init__(self, shape, step, sigma, clamp=True):
        self.shape = shape
        self.sigma = sigma
        self.step = step
        self.coldness = 20
        self.clamp = clamp

        self.kernel = lambda z: tf.exp(
            -2
            * (tf.clip_by_value(z - 0.5, clip_value_min=0, clip_value_max=np.inf) ** 2)
        )

        self.margin = self.sigma
        self.padding = 1 + math.ceil((self.margin + sigma) / step)
        self.radius = 1 + math.ceil(sigma / step)
        self.shape_in = [math.ceil(z / step) for z in self.shape]
        self.shape_mid = [
            z + 2 * self.padding - (2 * self.radius + 1) + 1 for z in self.shape_in
        ]
        self.shape_up = [self.step * z for z in self.shape_mid]
        self.shape_out = [z - step + 1 for z in self.shape_up]

        self.weight = np.zeros(
            (1, self.shape_out[0], self.shape_out[1], (2 * self.radius + 1) ** 2)
        )

        step_inv = [
            np.array(zm, dtype=np.float32) / np.array(zo, dtype=np.float32)
            for zm, zo in zip(self.shape_mid, self.shape_up)
        ]

        for ky in range(2 * self.radius + 1):
            for kx in range(2 * self.radius + 1):
                uy, ux = np.meshgrid(
                    np.arange(self.shape_out[0], dtype=np.float32),
                    np.arange(self.shape_out[1], dtype=np.float32),
                )
                iy = np.floor(step_inv[0] * uy) + ky - self.padding
                ix = np.floor(step_inv[1] * ux) + kx - self.padding

                delta = np.sqrt(
                    (uy - (self.margin + self.step * iy)) ** 2
                    + (ux - (self.margin + self.step * ix)) ** 2
                )

                k = ky * (2 * self.radius + 1) + kx

                self.weight[0, :, :, k] = self.kernel(delta / sigma)
        self.weight = tf.convert_to_tensor(self.weight, dtype=tf.float32)

    def generate(self, mask_in):
        paddings = [
            [0, 0],
            [self.padding, self.padding],
            [self.padding, self.padding],
            [0, 0],
        ]
        mask = tf.pad(mask_in, paddings)
        # NOTE(brendan): this unfolding is for convolution
        mask = tf.image.extract_patches(
            mask,
            sizes=(1, (2 * self.radius) + 1, (2 * self.radius) + 1, 1),
            strides=4 * (1,),
            rates=4 * (1,),
            padding="VALID",
        )
        mask = tf.image.resize(mask, size=self.shape_up, method="nearest")
        mask = mask[:, : -self.step + 1, : -self.step + 1, :]
        # NOTE(brendan): convolve Gaussian weights with mask (smoothness inductive bias)
        mask = self.weight * mask

        mask = mask * softmax(self.coldness * mask, axis=-1)
        mask = tf.reduce_sum(mask, axis=-1, keepdims=True)

        m = round(self.margin)
        if self.clamp:
            mask = tf.clip_by_value(mask, clip_value_min=0, clip_value_max=1)
        cropped = mask[:, m : m + self.shape[0], m : m + self.shape[1], :]
        return cropped, mask


def imsmooth_TF(
    tensor, sigma, stride=1, padding=0, padding_mode="constant", padding_value=0
):
    assert sigma >= 0
    width = math.ceil(4 * sigma)
    filt = np.arange(-width, width + 1) / (math.sqrt(2) * sigma + EPSILON_SINGLE)
    filt = tf.exp(-filt * filt)
    filt /= tf.reduce_sum(filt)
    filt = tf.cast(filt, tf.float32)
    num_channels = tensor.shape[-1]

    vertical_blur_filter = tf.reshape(filt, (-1, 1, 1, 1))
    vertical_blur_filter = tf.tile(vertical_blur_filter, (1, 1, num_channels, 1))
    x = tf.nn.depthwise_conv2d(
        tensor,
        vertical_blur_filter,
        strides=(1, 1, 1, 1),
        padding="SAME",
        data_format="NHWC",
    )
    horizontal_blur_filter = tf.reshape(filt, (1, -1, 1, 1))
    horizontal_blur_filter = tf.tile(horizontal_blur_filter, (1, 1, num_channels, 1))
    x = tf.nn.depthwise_conv2d(
        tensor,
        horizontal_blur_filter,
        strides=(1, 1, 1, 1),
        padding="SAME",
        data_format="NHWC",
    )
    return x


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
        w = tf.reshape(mask, (n, *mask.shape[1:]), 1)
        w = w * (self.num_levels - 1)
        k = tf.floor(w)
        w = w - k
        k = tf.cast(k, dtype=tf.int64)

        y = self.pyramid[None, :]
        y = tf.cast(y, tf.float32)
        assert n == 1

        k = tf.tile(k, multiples=(1, 1, 1, 3))
        k = k[..., tf.newaxis]
        first_idx = tf.zeros(k.shape, dtype=tf.int64)
        y_idx, x_idx, c_idx = tf.meshgrid(
            tf.range(y.shape[2], dtype=tf.int64),
            tf.range(y.shape[3], dtype=tf.int64),
            tf.range(3, dtype=tf.int64),
            indexing="ij",
        )
        xy_indices = tf.stack((y_idx, x_idx, c_idx), axis=-1)
        xy_indices = xy_indices[tf.newaxis]
        indices = tf.concat((first_idx, k, xy_indices), axis=-1)
        y0 = tf.gather_nd(y, indices=indices)

        k = tf.clip_by_value(
            k + 1, clip_value_min=0, clip_value_max=self.num_levels - 1
        )
        indices = tf.concat((first_idx, k, xy_indices), axis=-1)
        y1 = tf.gather_nd(y, indices=indices)

        return (1 - w) * y0 + w * y1

    def to(self, dev):
        self.pyramid.to(dev)
        return self


def task3_2():
    model_path = os.path.join("project_a_supp", "models", "HMT.h5")
    model = keras.models.load_model(model_path)
    # NOTE(brendan): remove softmax to optimize class logit (linear gradient)
    model.layers[-1].activation = None
    optimizer = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)

    test_dir = os.path.join("project_a_supp", "hmt_dataset", "HMT_test")
    hmt_test_datagen = ImageDataGenerator(rescale=1 / 255.0)
    input_img_size = (224, 224)
    test_generator = hmt_test_datagen.flow_from_directory(
        test_dir,
        class_mode="categorical",
        interpolation="bilinear",
        target_size=input_img_size,
        batch_size=1,
        shuffle=False,
    )

    # NOTE(brendan): Extremal perturbation hyperparameters
    areas = [0.1]
    initial_regul_weight = 300
    max_iter = 800
    sigma = 21
    num_levels = 8
    step = 7

    # NOTE(brendan): just to figure out correct/incorrect predictions for visualization
    correct_indices = []
    incorrect_indices = []
    num_examples = len(test_generator)
    for example_idx, (image_batch, label_batch) in enumerate(test_generator):
        if example_idx >= num_examples:
            break

        hmt_label = label_batch[0]
        hmt_prediction = model(image_batch).numpy()

        hmt_prediction = np.argmax(hmt_prediction)
        if hmt_prediction == np.argmax(hmt_label):
            correct_indices.append(example_idx)
            continue
        incorrect_indices.append(example_idx)
    print(f"correct: {len(correct_indices)} incorrect: {len(incorrect_indices)}")

    num_vis_examples = 4
    visualization_type = "correct"
    if visualization_type == "correct":
        example_indices = correct_indices
    else:
        example_indices = incorrect_indices
    test_generator.reset()
    num_visualized = 0
    drop_rate_cma = 0
    increase_rate_cma = 0
    for example_idx, (image_batch, label_batch) in enumerate(test_generator):
        # if example_idx not in example_indices:
        #     continue

        regul_weight = initial_regul_weight
        # NOTE(brendan): Extremal perturbation algorithm
        perturbation = Perturbation_TF(image_batch, num_levels=num_levels)
        mask_generator = MaskGenerator_TF(perturbation.pyramid.shape[1:3], step, sigma)

        # NOTE(brendan): Prepare reference area vector
        max_area = np.prod(mask_generator.shape_out)
        reference = np.ones((len(areas), max_area))
        for area_idx, area in enumerate(areas):
            reference[area_idx, : int(max_area * (1 - area))] = 0
        reference = tf.convert_to_tensor(reference, dtype=tf.float32)

        pmask = tf.ones((len(areas), *mask_generator.shape_in, 1))
        pmask = tf.Variable(pmask)
        y = model(image_batch)
        target_channel = np.argmax(tf.squeeze(y))
        for iter_t in range(max_iter):
            with tf.GradientTape() as tape:
                # NOTE(brendan): generate mask from smooth manifold
                mask_, mask = mask_generator.generate(pmask)

                # NOTE(brendan): use "preserve" variant of EP to perturb input
                # with smooth mask
                x = perturbation.apply(mask_)

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

        model.layers[-1].activation = softmax
        y = model(x)
        confidence_perturbed = y[:, target_channel]
        y = model(image_batch)
        confidence_raw = y[:, target_channel]
        drop_rate = (confidence_raw - confidence_perturbed) / (
            confidence_raw + EPSILON_SINGLE
        )
        increase_rate = int(confidence_perturbed > confidence_raw)
        drop_rate_cma = (drop_rate + (example_idx * drop_rate_cma)) / (example_idx + 1)
        increase_rate_cma = (increase_rate + (example_idx * increase_rate_cma)) / (
            example_idx + 1
        )
        print(f"Drop rate {drop_rate_cma}")
        print(f"Increase rate {increase_rate_cma}")
        model.layers[-1].activation = None
        # num_rows = 2
        # plt.subplot(num_rows, num_vis_examples, 1 + num_visualized)
        # plt.imshow(np.squeeze(image_batch))
        # plt.axis("off")
        # plt.title("Input")

        # plt.subplot(num_rows, num_vis_examples, 1 + num_vis_examples + num_visualized)
        # plt.imshow(np.squeeze(x.numpy()))
        # plt.axis("off")
        # plt.title("Perturbed")

        # num_visualized += 1
        # if num_visualized >= num_vis_examples:
        #     break
    plt.show()


if __name__ == "__main__":
    task3_2()
