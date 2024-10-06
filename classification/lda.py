from threading import Thread

import numpy as np
from numpy import dtype

from classification.classifier import GenerativeClassifier
from utils.constant import EPS
from utils.test_model import test_model_on_binary_mnist, test_model_on_mnist


class LinearDiscriminantAnalysisClassifier(GenerativeClassifier):

    def __init__(self):
        super().__init__()

        self.__means: np.ndarray[np.float64] = None
        self.__Sigma: np.ndarray[np.float64] = None
        self.__det_Sigma: np.float64 = None
        self.__Sigma_inv: np.ndarray[np.float64] = None
        self.__pointless_features: np.ndarray[np.int16] = None

    def train(self, training_examples_set: np.ndarray[np.float64], training_examples_label_set: np.ndarray[np.int16]):

        training_examples_set = training_examples_set.astype(np.float64)

        std_devs = np.std(training_examples_set, axis=0)
        self.__pointless_features = np.where(std_devs == 0)[0]

        if len(self.__pointless_features) > 0:
            print(f"These features will be ignored ({len(self.__pointless_features)}): {self.__pointless_features}")
            training_examples_set = np.delete(training_examples_set, self.__pointless_features, axis=1)

        training_examples_set = (training_examples_set - np.mean(training_examples_set, axis=0)) / np.std(training_examples_set, axis=0)

        N = len(training_examples_label_set)
        D = training_examples_set.shape[1]

        print(f"Number of features considered for each example: {D}")

        self._classes = np.unique(training_examples_label_set)
        self._classes_priors = np.empty(len(self._classes), dtype=np.float64)
        self.__means = np.empty((len(self._classes), D), dtype=np.float64)
        self.__Sigma = np.zeros((D, D), dtype=np.float64)

        for i, c in enumerate(self._classes):
            c_examples = training_examples_set[training_examples_label_set == c]

            N_of_c = c_examples.shape[0]

            self._classes_priors[i] = N_of_c / N

            mean = np.mean(c_examples, axis=0)

            self.__means[i] = mean

            for x in c_examples[:]:
                x -= mean
                self.__Sigma += x.reshape((D, 1)) @ x.reshape((1, D))

        self.__Sigma /= N
        self.__Sigma += EPS + np.eye(D, D)

        print(f"Sigma cond: {np.linalg.cond(self.__Sigma)}")

        self.__det_Sigma = np.linalg.det(self.__Sigma)

        assert self.__det_Sigma != 0, "Sigma is not invertible"

        self.__Sigma_inv = np.linalg.inv(self.__Sigma)


    def predict(self, input: np.ndarray[np.float64]) -> np.int16:


        if len(self.__pointless_features) > 0:
            input = np.delete(input, self.__pointless_features, axis=0)

        posteriors = np.empty(len(self._classes), dtype=np.float64)
        D = len(input)

        for i, c in enumerate(self._classes):

            posteriors[i] = np.log(self._classes_priors[i]) \
                            - 0.5 * D * np.log(2 * np.pi) \
                            - 0.5 * np.log(self.__det_Sigma) \
                            - 0.5 * (input - self.__means[i]).reshape((1, D)) @ self.__Sigma_inv @ (input - self.__means[i]).reshape((D, 1))

        return self._classes[np.argmax(posteriors)]

    def predict_set(self, input_set: np.ndarray[np.float64]) -> np.ndarray[np.int16]:
        N = input_set.shape[0]

        predictions = np.empty(N, dtype=np.float64)

        for i, input in enumerate(input_set[:]):
            predictions[i] = self.predict(input)

        return predictions

if __name__ == '__main__':
    test_model_on_mnist(LinearDiscriminantAnalysisClassifier())





