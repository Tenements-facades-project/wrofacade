import argparse
import os
import numpy as np
import cv2

from src.utils.grammars_utils.ascfg import Grammar
from src.utils.grammars_utils.bayesian_merging import BayesianMerger


parser = argparse.ArgumentParser()
parser.add_argument("imgs_dir")
parser.add_argument("masks_dir")
parser.add_argument("output_dir")
parser.add_argument("--output_pickle_name", default="grammar.pickle")
parser.add_argument("--checkpoints_dir", default="checkpoints")
parser.add_argument("--imgs_ext", default="png")

args = parser.parse_args()


def load_facade_and_mask(facade_name: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(os.path.join(
        args.imgs_dir, f"{facade_name}.{args.imgs_ext}"
    ))
    mask = cv2.imread(os.path.join(
        args.masks_dir, f"{facade_name}.png"
    ))
    return img, mask


img, mask = load_facade_and_mask('szczepin_podwale_7')
print(img.shape)
print(mask.shape)
