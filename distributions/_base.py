# Imports
from abc import ABC, abstractmethod

# Packages
import numpy as np

# --------------------------------------------------------------------


class abstractDistribution(ABC):
    @abstractmethod
    def generate(self, rng: np.random.Generator,
                 samples: int) -> np.ndarray:
        """
        Generates a number of random samples of the distribution.

        Parameters
        ----------
        rng: numpy.random.Generator object
            Pre-seed generator
        samples: int
            Number of samples to return

        Returns
        ----------
        x: numpy array of shape (n, m)
            where n is the number of samples and m is the number of features.

        Notes
        -------

        """
        pass

    @abstractmethod
    def logPdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the pdf at a set of points.

        Parameters
        ----------
        x: numpy array of shape (n, m)
            where n is the number of samples and m is the number of features.

        Returns
        ----------
        score: numpy array of shape (n)
            Equivalent to log(pdf(x))


        Notes
        -------

        """
        pass
