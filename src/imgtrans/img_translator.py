from abc import ABC, abstractmethod


class ImageTranslator(ABC):
    """Abstract class for image translation models"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass"""
        pass
