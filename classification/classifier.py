from abc import ABC, abstractmethod
import numpy as np


class Classifier(ABC):


    def __init__(self):
        self._classes: np.ndarray[np.int16] = None

    @property
    def classes(self) -> np.ndarray[np.int16]:
        return self._classes

    @abstractmethod
    def train(self, training_examples_set: np.ndarray[np.float64], training_examples_label_set: np.ndarray[np.int16]):
        pass

    @abstractmethod
    def predict(self, input: np.ndarray[np.float64]) -> np.int16:
        pass

    @abstractmethod
    def predict_set(self, input_set: np.ndarray[np.float64]) -> np.ndarray[np.int16]:
        pass


class GenerativeClassifier(Classifier, ABC):

    def __init__(self):
        super().__init__()

        self._classes_priors: np.ndarray[np.float64] = None


    @property
    def classes_priors(self) -> np.ndarray[np.float64]:
        return self._classes_priors