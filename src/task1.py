import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from mnist1d_utils import make_dataset


def task1():
    model_path = os.path.join("project_a_supp", "models", "MNIST1D.h5")
    model = keras.models.load_model(model_path)

    mnist1d = make_dataset()

    x_test = np.expand_dims(mnist1d["x_test"], axis=-1)
    y_test = mnist1d["y_test"]
    model.evaluate(x_test, y_test)

    num_classes = 10
    num_correct = 0
    num_correct_per_class = np.zeros(num_classes, dtype=np.int64)
    num_total_per_class = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(x_test)):
        digit_input = x_test[i : i + 1]
        digit_label = y_test[i : i + 1]
        digit_prediction = model(digit_input).numpy()
        digit_prediction = np.argmax(digit_prediction)
        if digit_prediction == digit_label:
            num_correct += 1
            num_correct_per_class[digit_label] += 1

        num_total_per_class[digit_label] += 1
    print(f"Accuracy: {num_correct/len(x_test)}")
    classwise_accuracy = num_correct_per_class / num_total_per_class
    print(f"Class-wise ccuracy: {classwise_accuracy}")

    plt.bar(x=range(num_classes), height=classwise_accuracy)
    plt.title("MNIST-1D Class-wise Accuracy")
    plt.xticks(range(10))
    plt.xlabel("Digit")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join("report", "images", "mnist1d-class-accuracy.png"), dpi=256)
    plt.clf()


if __name__ == "__main__":
    task1()
