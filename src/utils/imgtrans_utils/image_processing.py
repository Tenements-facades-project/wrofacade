import random
import torchvision.transforms as transforms
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def img_to_tensor(img: Image):
    img = img.resize((286, 286), Image.BICUBIC)
    img = transforms.ToTensor()(img)

    w_offset = random.randint(0, max(0, 286 - 256 - 1))
    h_offset = random.randint(0, max(0, 286 - 256 - 1))

    img = img[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

    img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def tensor_to_img(img_tens):
    return torch.clip(denorm(img_tens.detach().squeeze(0).cpu()).permute(1, 2, 0), 0, 255)
