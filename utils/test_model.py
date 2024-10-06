from classification.classifier import Classifier
import mnist
import numpy as np
from utils.confusion_matrix import plot_confusion_matrix




def test_model_on_binary_mnist(model: Classifier):

    mnist.DATASET_DIR = "../mnist"

    x_train, y_train, x_test, y_test = mnist.binary_mnist_sets(threshold=0.5)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    test_model(model, x_train, y_train, x_test, y_test)


def test_model_on_mnist(model: Classifier):

    mnist.DATASET_DIR = "../mnist"

    x_train, y_train, x_test, y_test = mnist.grayscale_mnist_sets()

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    test_model(model, x_train, y_train, x_test, y_test)


def test_model(model: Classifier, x_train, y_train, x_test, y_test):

    print(f"Training set -> number of examples: {len(x_train)}")
    print(f"Test set -> number of examples: {len(x_test)}")
    print('-' * 30)
    print(f'X -> shape: {x_train.shape}')
    print(f"X -> dtype: {x_train.dtype}")
    print(f"X -> min: {x_train.min()}")
    print(f"X -> max: {x_train.max()}")
    print(f"X -> values: {np.unique(x_train)}")
    print('-' * 30)
    print(f"Classes: {np.unique(y_train)}")

    # train
    print("start training...")
    model.train(x_train, y_train)
    print("training end")

    # test
    print("start test...")
    predictions = model.predict_set(x_test)

    # evaluate performances
    accuracy = np.sum(np.uint8(predictions == y_test)) / len(y_test)
    print(f'Accuracy: {accuracy * 100}%')

    plot_confusion_matrix(y_test, predictions, np.unique(y_train))
