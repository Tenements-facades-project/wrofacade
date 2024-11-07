from abc import ABC, abstractmethod


class FacadeGenerator(ABC):
    """Base class for all facade generators
    """

    @abstractmethod
    def generate_facade(self, params: dict = None):
        """
        Args:
            params -- dict containing parameters values for generating
        """
        pass
