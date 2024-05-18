from utils.general_utils.find_models import find_seg_model_using_name, find_trans_model_using_name
from config.hparam import hparam as hp

class SegmentAndTranslate:
    """Class for generating facades using the two modules: segmentation and image translation
    Args:
        segmentation_model_name -- name of the model used for segmentation, must be one of the subclasses of SegMaskGenerator
        image_translator_name -- name of the model used for image translation, must be one of the subclasses of ImageTranslator
    """

    def __init__(self,
                 segmentation_model_name: str = None,
                 translation_model_name: str = None,
                 ) -> None:

        # find model classes using their names
        segmentation_model_cls = find_seg_model_using_name(segmentation_model_name)
        image_translator_cls = find_trans_model_using_name(translation_model_name)
        # initialize the models
        self.segmentation_model = segmentation_model_cls(hp)
        self.image_translator = image_translator_cls(hp)

    def generate_facade(self, params = None):
        
        segmentated_image = self.segmentation_model.generate_mask(params) # return SegmentationMask class
        generated_facade = self.image_translator.pass_image(segmentated_image)

        return generated_facade
