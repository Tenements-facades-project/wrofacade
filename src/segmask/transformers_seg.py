from seg_mask import SegMaskGenerator
from transformers import SegformerForSemanticSegmentation
import torch
import os
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import numpy as np

class TransSegmentationHF(SegMaskGenerator):
    """seg_mask module implementation using segmentation model
    based on transformers with the use of hugging face interface

    Link to paper https://arxiv.org/abs/2105.15203
    """

    def __init__(self, model_name_or_path: str, local_files_only:bool = False, device: torch.device = None):
        """Initialize the TransSegmentationHF class.

        Parameters:
            model_name_or_path  -- string containing name or path to the model,
                          if string is an existing path the model is loaded from drive
                          if not the model will be looked up on Hugging 
                          Face platform

            local_files_only - boolean flag. If True the model wouldn't be looked
                          up and downloaded from Hugging face and instead
                          it would be looked up only locally
            
            device -- torch.device object. If not provided will use CUDA if avaliable,
                      if not CPU will be used
        """

        if device is None:
            device = torch.device('cuda')if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device 

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path = model_name_or_path, 
            local_files_only = local_files_only
            )

    def forward(self, image:Image):
        """Get a segmentation mask for the provided image.

        Parameters:
            image -- a PIL RGB image
        
        Returns:
            segmentation_mask - a segmentation mask object

        Raises:
            ValueError - if PIL image is not in RGB format
        """

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

        raise NotImplementedError()
        

