from src.segmask.seg_mask import SegMaskGenerator
from transformers import SegformerForSemanticSegmentation
import torch
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import numpy as np
from src.utils.segmentation_mask import SegmentationMask

class TransSegmentationHF(SegMaskGenerator):
    """seg_mask module implementation using segmentation model
    based on transformers with the use of hugging face interface

    Link to paper https://arxiv.org/abs/2105.15203
    """
        

    def __init__(self, hp: dict) -> None:
        """Initialize the TransSegmentationHF class.

        Parameters:
            hp - dictionary containing parametres with next entries:
                'model_name_or_path':str  -- string containing name or path to the model,
                            if string is an existing path the model is loaded from drive
                            if not the model will be looked up on Hugging 
                            Face platform
                
                'label2clr':dict[str,list] -- dictionary containing class labels and colours, keys' positional index
                            in the dict corresponds to the class id, e.g. {'window': [255,244,233]}

                'local_files_only':boolean - boolean flag. If True the model wouldn't be looked
                            up and downloaded from Hugging Face platform and instead
                            it would be looked up only locally
                
                'device':torch.device|str -- torch.device object. If not provided will use CUDA if avaliable,
                        if not CPU will be used
        """

        if hp['device'] is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            if isinstance(hp['device'], torch.device):
                self.device = hp['device'] 
            elif isinstance(hp['device'], str):
                self.device = torch.device(hp['device'])
            else:
                raise Exception('Wrong device name')

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path = hp['segmentator']['model_name_or_path'], 
            local_files_only = hp['segmentator']['local_files_only']
            )
        
        self.label2clr = hp['label2clr']

    def generate_mask(self, args: dict) -> SegmentationMask:
        """Get a segmentation mask for the provided image.

        Parameters:
            args -- dictionary containing next entries:
                'image':PIL.Image -- a PIL RGB image of a facade
        
        Returns:
            segmentation_mask - a segmentation mask object

        Raises:
            ValueError - if input PIL image is not in the RGB format
        """

        image = args['image']

        if ''.join(image.getbands()).lower() != 'rgb':
            raise ValueError('Image is not in RGB format')
        
        output = self.model((pil_to_tensor(image) / 255).unsqueeze(0).to(self.device))
        logits = output.logits

        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode='bilinear',
            align_corners=False
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        segmentation_mask = Image.fromarray(np.array(pred_seg.cpu()).astype(np.uint8))

        segmentation_mask = SegmentationMask(mask_img=segmentation_mask, label2clr=self.label2clr)

        return segmentation_mask
        

