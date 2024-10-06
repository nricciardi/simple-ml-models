import numpy as np


class NaiveBayesClassifier:

    def __init__(self):
        self.__classes: np.ndarray[np.int64] = None
        self.__classes_priors: np.ndarray[np.float64] = None
        self.__classes_likelihood: np.ndarray[np.float64] = None


    @property
    def classes_priors(self) -> np.ndarray[np.float64]:
        return self.__classes_priors

    @property
    def classes_likelihood(self) -> np.ndarray[np.float64]:
        return self.__classes_likelihood

    def fit(self, X: np.ndarray[np.int64], Y: np.ndarray[np.int8]):
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

    def predict(self, x: np.ndarray[np.int64]) -> np.int8:

        predictions = np.empty(len(self.__classes), dtype=np.int8)

        for i, c in enumerate(self.__classes):
            prediction = np.log10(self.__classes_priors[c] + EPS) + \
                         np.sum(np.log10(np.where(
                             x == 1,
                             self.__classes_likelihood[c] + EPS,
                             1 - self.__classes_likelihood[c]
                         )))

            predictions[i] = prediction

        predicted_class = self.__classes[np.argmax(predictions)]

        return predicted_class

    def predict_set(self, X: np.ndarray[np.int64]) -> np.ndarray[np.int8]:
        """
        Performs inference on test data.
        Inference is performed according to the Bayes rule:
        P = argmax_Y (log(P(X/Y)) + log(P(Y)) - log(P(X)))

        Parameters
        ----------
        X: np.ndarray
            MNIST test images. Has shape (n_test_samples, h, w).

        Returns
        -------
        prediction: np.ndarray
            model predictions over X. Has shape (n_test_samples,)
        """

        N = X.shape[0]

        predictions = np.empty(N, dtype=np.int8)

        for i, x in enumerate(X[:]):
            predictions[i] = self.predict(x)

        return predictions




def main():
    x_train, y_train, x_test, y_test, label_dict = load_mnist(threshold=0.5)

    print(f"Training set -> number of examples: {len(x_train)}")
    print(f"Test set -> number of examples: {len(x_test)}")
    print('-' * 30)
    print(f'X -> shape: {x_train.shape}')
    print(f"X -> dtype: {x_train.dtype}")
    print(f"X -> min: {x_train.min()}")
    print(f"X -> max: {x_train.max()}")
    print(f"X -> values: {np.unique(x_train)}")
    print('-' * 30)
    print(f"Classes: {(np.unique(y_train))}")

    nbc = NaiveBayesClassifier()

    # train
    nbc.fit(x_train, y_train)

    # test
    predictions = nbc.predict(x_test)     # x_test.reshape((len(x_test), -1))

    # evaluate performances
    accuracy = np.sum(np.uint8(predictions == y_test)) / len(y_test)
    print('Accuracy: {}'.format(accuracy))

if __name__ == '__main__':
    main()