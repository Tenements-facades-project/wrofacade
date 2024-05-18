from abc import ABC, abstractmethod
from imgtrans.img_translator import ImageTranslator
from imgtrans.pix2pix import Pix2PixModel
from segmask.seg_mask import SegMaskGenerator
from config.hparam import hparam as hp

class FacadeGenerator():
    """Class for FacadeGenerator.
    Class takes model names as arguments, and depending on the model provided builds a generation pipeline.
    
    """

    def __init__(self,
                 segmentation_model: str = None,
                 translation_model:str = None,
                 ) -> None:

        self.segmentation_model = segmentation_model()
        self.image_translator = translation_model(hp)

        pass

    @abstractmethod
    def generate_facade(self):
        pass