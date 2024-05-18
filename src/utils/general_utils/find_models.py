import importlib
import os
print(os.getcwd())
from src.imgtrans.img_translator import ImageTranslator
from src.segmask.seg_mask import SegMaskGenerator
from typing import Type

def find_seg_model_using_name(name: str = "transformers_seg") -> Type[SegMaskGenerator]:
    """Class for loading segmentation model from model name"""
    model_filename = "segmask." + name 
    modellib = importlib.import_module(model_filename)
    model = None
    for name, cls in modellib.__dict__.items():
        try:
            if issubclass(cls, SegMaskGenerator):
                model = cls
        except TypeError:
            pass
    if model is  None:
        raise KeyError("Model not found")
    return model

def find_trans_model_using_name(name: str = "pix2pix") -> Type[ImageTranslator]:
    """Class for loading translation model from model name"""
    model_filename = "src.imgtrans." + name 
    modellib = importlib.import_module(model_filename)
    model = None
    for name, cls in modellib.__dict__.items():
        try:
            if issubclass(cls, ImageTranslator):
                model = cls
        except TypeError:
             pass
    if model is  None:
        raise KeyError("Model not found")
    return model

if __name__ == "__main__":
    model = find_trans_model_using_name()
    print(model)