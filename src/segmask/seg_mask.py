from abc import ABC, abstractmethod


class SegMaskGenerator(ABC):
    """Base class for all mask generators
    (i.e. generates some segmentation mask).
    Such a generated mask can be translated into
    a new facade image

    Note that mask generation can be parametrized,
    and a parameter can be e.g. a facade image.
    """

    @abstractmethod
    def generate_mask(self, args: dict):
        pass
