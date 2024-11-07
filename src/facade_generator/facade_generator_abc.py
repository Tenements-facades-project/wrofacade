from abc import ABC, abstractmethod


class FacadeGenerator(ABC):
    """Base class for all facade generators
    """

    @abstractmethod
    def generate_facade(self, *args, **kwargs):
        pass
