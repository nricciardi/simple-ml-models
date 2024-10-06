import numpy as np
import os


MNIST_ZIP_FILE = "./mnist/mnist.zip"
OUTPUT_DIR = "./mnist"
DATASET_DIR = OUTPUT_DIR


def unzip_datasets(zip_file):
    import zipfile

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)

def binary_mnist_sets(threshold: float = 0.5) -> (np.ndarray[np.int8], np.ndarray[np.int8], np.ndarray[np.int8], np.ndarray[np.int8]):

    x_train, y_train, x_test, y_test = grayscale_mnist_sets()

    x_train = np.float32(x_train) / 255.
    x_train[x_train >= threshold] = 1
    x_train[x_train < threshold] = 0

    x_test = np.float32(x_test) / 255.
    x_test[x_test >= threshold] = 1
    x_test[x_test < threshold] = 0

    return x_train, y_train, x_test, y_test


def grayscale_mnist_sets() -> (np.ndarray[np.int8], np.ndarray[np.int8], np.ndarray[np.int8], np.ndarray[np.int8]):

    if not os.path.exists(f'{DATASET_DIR}/x_train.npy') \
        or not os.path.exists(f'{DATASET_DIR}/y_train.npy') \
        or not os.path.exists(f'{DATASET_DIR}/x_test.npy') \
        or not os.path.exists(f'{DATASET_DIR}/y_test.npy'):

        unzip_datasets(MNIST_ZIP_FILE)

    x_train = np.load(f'{DATASET_DIR}/x_train.npy')
    y_train = np.load(f'{DATASET_DIR}/y_train.npy')

    x_test = np.load(f'{DATASET_DIR}/x_test.npy')
    y_test = np.load(f'{DATASET_DIR}/y_test.npy')

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    unzip_datasets(MNIST_ZIP_FILE)