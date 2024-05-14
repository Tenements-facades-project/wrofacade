import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

from src.utils.grammars_utils.ascfg import Grammar
from src.utils.grammars_utils.bayesian_merging import BayesianMerger
from src.utils.lattice_utils.data_manipulations import resize_facade, crop_facade
from src.utils.lattice_utils.split_lines import infer_split_lines
from src.utils.lattice_utils.lattice import Lattice


parser = argparse.ArgumentParser()
parser.add_argument("imgs_dir")
parser.add_argument("masks_dir")
parser.add_argument("output_dir")
parser.add_argument("--output_pickle_name", default="grammar.pickle")
parser.add_argument("--checkpoints_dir", default="checkpoints")
parser.add_argument("--imgs_ext", default="png")
parser.add_argument("--downsampling_factor", default=0.17, type=float)
parser.add_argument("--background_label", default=0, type=int)

args = parser.parse_args()


def load_facade_and_mask(facade_name: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(os.path.join(args.imgs_dir, f"{facade_name}.{args.imgs_ext}"))
    mask = cv2.imread(os.path.join(args.masks_dir, f"{facade_name}.png"))
    if img is None:
        raise ValueError(
            f"There's no image of facade {facade_name} in provided imgs directory"
        )
    if mask is None:
        raise ValueError(
            f"There's no mask of facade {facade_name} in provided masks directory"
        )
    return img, mask[:, :, 2]


def remove_margin(l: list[int], left_margin: int, right_margin: int) -> list[int]:
    return [x for x in l if ((x > left_margin) and (x < right_margin))]


def get_facade_lattice(
    img: np.ndarray,
    mask: np.ndarray,
    dbscan_eps: int,
    dbscan_min_samples: int,
    margin: int,
    min_change_coef: float,
) -> Lattice:
    hor_lines_inds, ver_lines_inds = infer_split_lines(
        mask=mask,
        dbscan_params={"eps": dbscan_eps, "min_samples": dbscan_min_samples},
        min_change_coef=min_change_coef,
    )
    hor_lines_inds = remove_margin(hor_lines_inds, margin, mask.shape[0] - margin)
    ver_lines_inds = remove_margin(ver_lines_inds, margin, mask.shape[1] - margin)
    return Lattice.from_lines(
        img=img,
        mask=mask,
        horizontal_lines_inds=hor_lines_inds,
        vertical_lines_inds=ver_lines_inds,
    )


if __name__ == "__main__":

    # load facades' images and masks

    print("Loading images and masks of facades ...")
    facades_names = [os.path.splitext(fname)[0] for fname in os.listdir(args.imgs_dir)]
    facades = [load_facade_and_mask(facade_name) for facade_name in tqdm(facades_names)]
    print(f"Loaded {len(facades)} facades")

    # downsample facade's images and masks

    print("Downsampling images and masks ...")
    downsampled_facades: list[tuple[np.ndarray, np.ndarray]] = []
    for img, mask in tqdm(facades):
        downsampled_facades.append(
            resize_facade(img, mask, resizing_factor=args.downsampling_factor)
        )

    # crop facades

    print("Cropping facades ...")
    # create rectangular lattices just for cropping
    lattices = [
        get_facade_lattice(
            img=img,
            mask=mask,
            dbscan_eps=3,
            dbscan_min_samples=1,
            min_change_coef=0.02,
            margin=0,
        )
        for img, mask in downsampled_facades
    ]
    # crop facades' lattices
    lattices = [
        crop_facade(
            lattice=lattice,
            background_label=args.background_label,
        )
        for lattice in lattices
    ]
    # create images and masks from cropped lattices
    cropped_facades = [lattice.assemble_lattice() for lattice in lattices]

    # create final lattices from cropped images and masks

    print("Creating rectangular lattices ...")
    lattices = [
        get_facade_lattice(
            img=img,
            mask=mask,
            dbscan_eps=5,
            dbscan_min_samples=1,
            min_change_coef=0.15,
            margin=10,
        )
        for img, mask in cropped_facades
    ]
