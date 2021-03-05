import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def task3_1():
    model_path = os.path.join("project_a_supp", "models", "HMT.h5")
    model = keras.models.load_model(model_path)

    test_dir = os.path.join("project_a_supp", "hmt_dataset", "HMT_test")
    hmt_test_datagen = ImageDataGenerator(rescale=1 / 255.0)
    test_generator = hmt_test_datagen.flow_from_directory(
        test_dir,
        class_mode="categorical",
        interpolation="bilinear",
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
    )

    num_classes = 8
    num_correct_per_class = np.zeros(num_classes, dtype=np.int64)
    num_total_per_class = np.zeros(num_classes, dtype=np.int64)
    y_predicted_scores = []
    num_examples = len(test_generator)
    y_test_onehot = np.zeros((num_examples, num_classes), dtype=np.int64)
    y_test = np.zeros(num_examples, dtype=np.int64)
    for example_idx, (image_batch, label_batch) in enumerate(test_generator):
        if example_idx >= num_examples:
            break

        assert (image_batch.shape[0] == 1) and (label_batch.shape[0] == 1)
        image = image_batch[0]
        label = label_batch[0]
        label = np.argmax(label)

        y = model(image_batch).numpy()
        class_pred = np.argmax(y, axis=1).item()

        y_predicted_scores.append(y)

        num_total_per_class[label] += 1
        if class_pred == label:
            num_correct_per_class[label] += 1
        y_test_onehot[example_idx, label] = 1
        y_test[example_idx] = label
    num_true_positives = num_correct_per_class.sum()

    y_predicted_scores = np.concatenate(y_predicted_scores, axis=0)
    print(f"Accuracy: {num_true_positives/num_examples}")
    classwise_accuracy = num_correct_per_class / num_total_per_class
    print(f"Class-wise ccuracy: {classwise_accuracy}")

    plt.bar(x=range(num_classes), height=classwise_accuracy)
    plt.title("HMT Class-wise Accuracy")
    plt.xticks(range(num_classes))
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join("report", "images", "hmt-class-accuracy.png"), dpi=256)
    plt.clf()

    for i in range(num_classes):
        false_pos_rate, true_pos_rate, _ = roc_curve(
            y_test_onehot[:, i], y_predicted_scores[:, i]
        )
        digit_auc = auc(false_pos_rate, true_pos_rate)
        plt.plot(false_pos_rate, true_pos_rate, label=f"{i} (AUC = {digit_auc:.4f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("HMT ROC-AUC Curves")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join("report", "images", "hmt-roc-curve.png"), dpi=256)
    plt.clf()

    y_predicted_digits = np.argmax(y_predicted_scores, axis=1)
    digits_confusion_mtx = confusion_matrix(
        y_predicted_digits, y_test, normalize="true"
    )
    plt.title("HMT Confusion Matrix")
    disp = ConfusionMatrixDisplay(digits_confusion_mtx)
    disp.plot()
    # plt.savefig(os.path.join("report", "images", "hmt-confusion-matrix"), dpi=256)
    plt.show()

    for i in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(
            y_test_onehot[:, i], y_predicted_scores[:, i]
        )
        avg_precision = average_precision_score(
            y_test_onehot[:, i], y_predicted_scores[:, i]
        )
        f1 = f1_score(y_test_onehot[:, i], np.round(y_predicted_scores[:, i]))
        plt.plot(
            recall, precision, label=f"{i} (AP = {avg_precision:.3f}, F1 = {f1:.2f})"
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("HMT Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join("report", "images", "hmt-precision-recall.png"), dpi=256)
    plt.clf()


if __name__ == "__main__":
    task3_1()
