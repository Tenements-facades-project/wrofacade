from PIL import Image
import numpy as np
from collections import Counter

class SegmentationMask:
    """Class containing segmentation mask and class info
    Args:
        mask_img -- grayscale PIL.Image 
        label2clr -- dictionary containing class labels and colours, key's positional index
                     in the dict corresponds to the class id, e.g. {'window': [255,244,233]}
    """
    def __init__(self, 
                 mask_img: Image,
                 label2clr: dict[str, list[int]]) -> None:
        self.mask = mask_img
        self.label2clr = label2clr

        self.check_colours()
        self.check_mask()

    def check_mask(self) -> None:
        """Checkes if the self.mask is a valid PIL.Image object

        Raises:
            ValueError: If self.mask has more than 1 channel

            ValueError: If self.mask contains unknown class ids
        """

        if len(self.mask.getbands()) != 1:
            raise ValueError("Segmentation mask should be a 1-channel image")
        
        # check if labels are ok
        if any(x not in range(len(self.label2clr)) for x in np.unique(np.array(self.mask))):
            raise ValueError("Mask contains unknown class ids")
        
    def check_colours(self) -> None:
        """Checkes if the self.label2clr has unique values

        Raises:
            ValueError: If keys (class labels) have duplicates

            ValueError: If values (class colours) have duplicates
        """
        
        duplicated_colours = []

        for colour_code in self.label2clr.values():
            if not any([colour_code == c for c in duplicated_colours]):
                occurences = sum([colour_code == c for c in self.label2clr.values()])
                if occurences > 1:
                    duplicated_colours.append(colour_code)
        
        if len(duplicated_colours) > 0:
            raise ValueError(f"The following colours are duplicated: {duplicated_colours}")
        
        
    def forward(self) -> Image:
        """Returns a copy of PIL.Image containing pixels' class ids

        Returns:
            PIL.Image object containing assigned class ids for each pixel
        """
        return self.mask.copy() 

    def get_rgb_mask(self) -> Image:
        """Creates a PIL.Image containing RGB segmentation mask with class 
        assigned colours. If provided colours have less than 3 channels 
        they will be padded with zeros automatically

        Returns:
            PIL.Image object containing RGB segmentation mask with class assigned colours
        """
        colour_list = []

        for k, class_colour in self.label2clr.items():
            cc = class_colour

            if len(cc) > 3:
                raise ValueError(f'Class colour has more than 3 channels: {k} - {cc}')
            
            if len(cc) == 0:
                raise ValueError(f'Class colour is not provided: {k} - {cc}')
            
            if len(cc) < 3:
                cc += [0] * (3 - len(cc))

            colour_list.append(cc)
        
        np_mask = np.array(self.mask)

        color_seg = np.zeros((np_mask.shape[0], np_mask.shape[1], 3), dtype=np.uint8)
        palette = np.array(colour_list)
        for label, color in enumerate(palette):
            color_seg[np_mask == label, :] = color

        image = Image.fromarray(color_seg, mode='RGB')

        return image