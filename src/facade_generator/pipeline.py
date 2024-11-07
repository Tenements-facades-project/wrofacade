from src.utils.general_utils.find_models import (
    find_seg_model_using_name,
    find_trans_model_using_name,
)
from src.config.hparam import hparam as hp
from src.facade_generator.facade_generator_abc import FacadeGenerator


class SegmentAndTranslate(FacadeGenerator):
    """Class for generating facades using the two modules: segmentation and image translation
    Args:
        segmentation_model_name -- name of the model used for segmentation, must be one of the subclasses of SegMaskGenerator
        image_translator_name -- name of the model used for image translation, must be one of the subclasses of ImageTranslator
        return_mask: whether to return source segmentation mask along with generated image
    """

    def __init__(
        self,
        segmentation_model_name: str = None,
        translation_model_name: str = None,
        return_mask: bool = False,
    ) -> None:

        # find model classes using their names
        segmentation_model_cls = find_seg_model_using_name(segmentation_model_name)
        image_translator_cls = find_trans_model_using_name(translation_model_name)
        # initialize the models
        self.segmentation_model = segmentation_model_cls(hp)
        self.image_translator = image_translator_cls(hp)
        self.return_mask: bool = return_mask

    def generate_facade(self, params: dict = None):
        segmentated_image = self.segmentation_model.generate_mask(
            params
        )  # return SegmentationMask class
        generated_facade = self.image_translator.pass_image(segmentated_image)

        if self.return_mask:
            return segmentated_image, generated_facade
        return generated_facade
