import argparse
import os
from collections import OrderedDict
import torch

from utils.imgtrans_utils import networks
from utils.imgtrans_utils.image_processing import img_to_tensor, tensor_to_img
from img_translator import ImageTranslator
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
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.translator.input_nc, opt.translator.output_nc, opt.translator.ngf, opt.translator.netG, opt.translator.norm,
                                      not opt.translator.no_dropout, opt.translator.init_type, opt.translator.init_gain, self.device)

        if self.isTrain:  # define a discriminator
            self.netD = networks.define_D(opt.translator.input_nc + opt.translator.output_nc, opt.translator.ndf, opt.translator.netD,
                                          opt.translator.n_layers_D, opt.translator.norm, opt.translator.init_type, opt.translator.init_gain, self.device)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.translator.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.translator.lr, betas=(opt.translator.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.translator.lr, betas=(opt.translator.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.translator.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def pass_image(self, img):
        """Generates fake image given a tensor
        Parameters:
            img (PIL.Image) -- image to translate 
        """
        # converting image to tensor
        img = img_to_tensor(img)
        img = img.unsqueeze(0)
        # generating new image
        fake_image = self.netG(img.to(self.device))
        out = tensor_to_img(fake_image)
        out.to(self.device)
        return out