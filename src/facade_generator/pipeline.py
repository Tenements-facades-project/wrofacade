from utils.general_utils.find_models import find_seg_model_using_name, find_trans_model_using_name
from config.hparam import hparam as hp

class SegmentAndTranslate:
    """Class for generating facades using the two modules: segmentation and image translation
    Args:
        segmentation_model --
        image_translator -- 
    """

    def __init__(self,
                 segmentation_model: str = None,
                 translation_model: str = None,
                 ) -> None:

        # find model classes using their names
        segmentation_model_cls = find_seg_model_using_name(segmentation_model)
        image_translator_cls = find_trans_model_using_name(translation_model)
        # initialize the models
        self.segmentation_model = segmentation_model_cls(hp)
        self.image_translator = image_translator_cls(hp)

    def generate_facade(self):
        pass