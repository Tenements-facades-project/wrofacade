
import torch
from src.utils.stack_gan_utils.config import cfg, cfg_from_file
from src.utils.stack_gan_utils.model import G_NET
from src.facade_generator.gan_generator import GANGeneratorTorch
from src.config.hparam import Hparam
import pytest
from pathlib import Path


class TestGANtorch:
    
    def test_stackgan_load(self):
        
        model = G_NET()

        model = torch.nn.DataParallel(model, range(1))

        cfg_from_file('checkpoints/stackGan/facade_3stages_color.yml')

        hp = Hparam()
        hp.checkpoint_path = 'checkpoints/stackGan/netG_56500.pth'
        hp.input_dim = 100
        hp.distribution = "normal"

        gan_generator = GANGeneratorTorch(model=model, hp=hp)

        assert True
    
    def test_stackgan_generate(self):
        
        model = G_NET()

        model = torch.nn.DataParallel(model, range(1))

        cfg_from_file('checkpoints/stackGan/facade_3stages_color.yml')

        hp = Hparam()
        hp.checkpoint_path = 'checkpoints/stackGan/netG_56500.pth'
        hp.input_dim = 100
        hp.distribution = "normal"

        gan_generator = GANGeneratorTorch(model=model, 
                                          hp=hp,
                                          forward_decorator=lambda x: x[0][2])

        result = gan_generator.generate_facade()

        assert list(result.shape) == [256, 256, 3]



