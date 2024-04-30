from PIL import Image
import json
import numpy as np

class SegmentationMask:
    """Class containing segmentation mask
    Args:
        mask_img -- grayscale PIL.Image 
        labels -- dictionary containing class labels, e.g. {1: 'window'}
    """
    def __init__(self, 
                 mask_img: Image,
                 labels: dict) -> None:
        self.mask = mask_img
        self.check_mask()
        self.labels = labels

    def check_mask(self):
        if len(self.mask.getbands()) != 1:
            raise ValueError("Segmentation mask should be a 1-channel image")
        
        # check if labels are ok
        if any(x not in self.labels for x in np.unique(np.array(self.mask))):
            raise ValueError("Mask containing invalid labels")

    def convert_to_rgb(self):
        raise NotImplementedError