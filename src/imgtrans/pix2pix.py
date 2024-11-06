"""
Code based on original Pix2Pix paper
Source repo available via https://github.com/phillipi/pix2pix
"""

import argparse
import os
from collections import OrderedDict
import torch

from src.utils.imgtrans_utils import networks
from src.utils.imgtrans_utils.image_processing import img_to_tensor, tensor_to_img
from .img_translator import ImageTranslator
from src.utils.segmentation_mask import SegmentationMask
import matplotlib.pyplot as plt
from PIL import Image

# from image_processing import img_to_tensor, tensor_to_img

class Pix2PixModel(ImageTranslator):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        ImageTranslator.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define networks
        self.netG = networks.define_G(opt.translator.input_nc, opt.translator.output_nc, opt.translator.ngf, opt.translator.netG, opt.translator.norm,
                                      not opt.translator.no_dropout, opt.translator.init_type, opt.translator.init_gain, self.device)
        self.load_networks(opt.translator.load_epoch)
        self.eval()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.translator.direction == 'AtoB'
        self.real_A = (input[0] if AtoB else input[1]).to(self.device)
        self.real_B = (input[1] if AtoB else input[0]).to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def pass_image(self, img: SegmentationMask):
        """Generates fake image given a tensor
        Parameters:
            img (SegmentationMask) -- image to translate 
        """
        # converting image to tensor
        img = img.get_rgb_mask()
        img = img_to_tensor(img, self.opt)
        self.real_A = img.unsqueeze(0)

        # generating new image
        self.forward()
        out = tensor_to_img(self.fake_B)
        return out
