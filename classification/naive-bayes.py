import numpy as np
from classification.classifier import GenerativeClassifier
from utils.constant import EPS
from utils.test_model import test_model_on_binary_mnist


class NaiveBayesClassifier(GenerativeClassifier):

    def __init__(self):
        super().__init__()

        self.__classes_likelihood: np.ndarray[np.float64] = None

    @property
    def classes_likelihood(self) -> np.ndarray[np.float64]:
        return self.__classes_likelihood

    def train(self, X: np.ndarray[np.float64], Y: np.ndarray[np.int16]):
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

        self._classes = np.unique(Y)
        self._classes_priors = np.empty(self._classes.shape[0])
        self.__classes_likelihood = np.empty((self._classes.shape[0], X.shape[1]))

        N = Y.shape[0]

        for c in self._classes:
            X_of_c: np.ndarray = X[Y == c]

            self._classes_priors[c] = X_of_c.shape[0] / N

            self.__classes_likelihood[c] = X_of_c.mean(0)

    def predict(self, input: np.ndarray[np.float64]) -> np.int16:

        classes_probabilities = np.empty(len(self._classes), dtype=np.float64)

        for i, c in enumerate(self._classes):
            class_probability = np.log10(self._classes_priors[c] + EPS) + \
                         np.sum(np.log10(np.where(
                             input == 1,
                             self.__classes_likelihood[c] + EPS,
                             1 - self.__classes_likelihood[c]
                         )))

            classes_probabilities[i] = class_probability

        predicted_class = self._classes[np.argmax(classes_probabilities)]

        return predicted_class

    def predict_set(self, test_set: np.ndarray[np.float64]) -> np.ndarray[np.int16]:

        N = test_set.shape[0]

        predictions = np.empty(N, dtype=np.float64)

        for i, input in enumerate(test_set[:]):
            predictions[i] = self.predict(input)

        return predictions


if __name__ == '__main__':
    test_model_on_binary_mnist(NaiveBayesClassifier())