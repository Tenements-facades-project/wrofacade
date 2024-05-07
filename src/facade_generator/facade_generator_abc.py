from abc import ABC, abstractmethod
from imgtrans.img_translator import ImageTranslator
from imgtrans.pix2pix import Pix2PixModel
from segmask.seg_mask import SegMaskGenerator
from config.hparam import hparam as hp

class FacadeGenerator:
    """Class for FacadeGenerator."""

    def __init__(self) -> None:

        self.image_translator = Pix2PixModel(hp)
        pass
