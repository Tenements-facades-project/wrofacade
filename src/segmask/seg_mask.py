from abc import ABC, abstractmethod


class SegMaskGenerator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_mask(self, args:dict):
        pass
