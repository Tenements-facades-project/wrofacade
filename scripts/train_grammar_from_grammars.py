"""This scripts loads several stochastic grammars
from files, merges them into one grammar and then trains
it using provided images and their segmentation masks
"""

import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm

from src.utils.grammars_utils.ascfg import merge_grammars
from src.utils.grammars_utils.bayesian_merging import BayesianMerger
from src.utils.lattice_utils.data_manipulations import resize_facade, crop_facade
from src.utils.grammars_utils.training_utils import (
    load_facade_and_mask,
    get_facade_lattice,
)
from src.utils.dataset_metadata import SEGFACADEDataset


parser = argparse.ArgumentParser()
parser.add_argument("imgs_dir")
parser.add_argument("masks_dir")
parser.add_argument("input_grammars_dir")
parser.add_argument("output_dir")
parser.add_argument("--window_labels", nargs="*", type=int, required=True)
parser.add_argument("--output_pickle_name", default="grammar.pickle")
parser.add_argument("--checkpoints_dir", default="checkpoints")
parser.add_argument("--checkpoint_every_n", type=int, default=100)
parser.add_argument("--downsampling_factor", default=0.17, type=float)
parser.add_argument("--background_label", default=0, type=int)
parser.add_argument("--n_random_draws", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=500)

args = parser.parse_args()


if __name__ == "__main__":

    # load images and masks
    facades: list[tuple[np.ndarray, np.ndarray]] = []
    facades_names = [os.path.splitext(fname)[0] for fname in os.listdir(args.imgs_dir)]
    facades += [
        load_facade_and_mask(
            facade_name=facade_name, imgs_dir=args.imgs_dir, masks_dir=args.masks_dir
        )
        for facade_name in tqdm(facades_names)
    ]
    print(f"Loaded f{len(facades)} facades")

    # downsample facade's images and masks

    print("Downsampling images and masks ...")
    downsampled_facades: list[tuple[np.ndarray, np.ndarray]] = []
    for img, mask in tqdm(facades):
        downsampled_facades.append(
            resize_facade(img, mask, resizing_factor=args.downsampling_factor)
        )

    # convert masks images to labels

    print("Masks converting ...")
    metadata = SEGFACADEDataset()
    downsampled_facades = [
        (img, metadata.parse_segmentation_image(mask))
        for img, mask in tqdm(downsampled_facades)
    ]

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

    # load input grammars
    print("Input grammars loading")
    input_grammars = []
    for path in tqdm(os.listdir(args.input_grammars_dir)):
        with open(path, "rb") as f:
            input_grammars.append(pickle.load(f))

    # merge into one grammar
    grammar = merge_grammars(input_grammars)

    # train
    print("Training")
    merger = BayesianMerger(n_random_draws=80, w=2.0, strategy="random_draw")
    grammar, loss_val = merger.perform_merging(
        grammar=grammar,
        lattices=lattices,
        n_epochs=100,
        checkpoint_dir="checkpoints_final",
        checkpoint_every_n=5,
    )

    with open("final_grammar.pickle", "wb") as f:
        pickle.dump(grammar, f)
