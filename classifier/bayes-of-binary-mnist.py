import numpy as np
import mnist
import utils.confusion_matrix
from utils.constant import EPS
from utils.confusion_matrix import plot_confusion_matrix


class NaiveBayesClassifier:

    def __init__(self):
        self.__classes: np.ndarray[np.int8] = None
        self.__classes_priors: np.ndarray[np.float64] = None
        self.__classes_likelihood: np.ndarray[np.float64] = None


    @property
    def classes_priors(self) -> np.ndarray[np.float64]:
        return self.__classes_priors

    @property
    def classes_likelihood(self) -> np.ndarray[np.float64]:
        return self.__classes_likelihood

    @property
    def classes(self) -> np.ndarray[np.int64]:
        return self.__classes

    def fit(self, X: np.ndarray[np.int64], Y: np.ndarray[np.int64]):
        """
        Computes, for each class, a naive likelihood model (self._pixel_probs_given_class),
        and a prior probability (self.class_priors).
        Both quantities are estimated from examples X and Y.

        Parameters
        ----------
        X: np.array
            input MNIST digits. Has shape (n_train_samples, h, w)
        Y: np.array
            labels for MNIST digits. Has shape (n_train_samples,)
        """

        self.__classes = np.unique(Y)
        self.__classes_priors = np.empty(self.__classes.shape[0])
        self.__classes_likelihood = np.empty((self.__classes.shape[0], X.shape[1], X.shape[2]))

        N = Y.shape[0]

        for c in self.__classes:
            X_of_c: np.ndarray = X[Y == c]

            self.__classes_priors[c] = X_of_c.shape[0] / N

            self.__classes_likelihood[c] = X_of_c.mean(0)

    def predict(self, input: np.ndarray[np.int64]) -> np.int64:

        classes_probabilities = np.empty(len(self.__classes), dtype=np.float64)

        for i, c in enumerate(self.__classes):
            class_probability = np.log10(self.__classes_priors[c] + EPS) + \
                         np.sum(np.log10(np.where(
                             input == 1,
                             self.__classes_likelihood[c] + EPS,
                             1 - self.__classes_likelihood[c]
                         )))

            classes_probabilities[i] = class_probability

        predicted_class = self.__classes[np.argmax(classes_probabilities)]

        return predicted_class

    def predict_set(self, test_set: np.ndarray[np.int64]) -> np.ndarray[np.int64]:

        N = test_set.shape[0]

        predictions = np.empty(N, dtype=np.int64)

        for i, input in enumerate(test_set[:]):
            predictions[i] = self.predict(input)

        return predictions




def main():

    mnist.DATASET_DIR = "../mnist"

    x_train, y_train, x_test, y_test = mnist.binary_mnist_sets(threshold=0.5)

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

    nbc = NaiveBayesClassifier()

    # train
    nbc.fit(x_train, y_train)

    # test
    predictions = nbc.predict_set(x_test)

    # evaluate performances
    accuracy = np.sum(np.uint8(predictions == y_test)) / len(y_test)
    print(f'Accuracy: {accuracy * 100}%')

    plot_confusion_matrix(y_test, predictions, np.unique(y_train))

if __name__ == '__main__':
    main()