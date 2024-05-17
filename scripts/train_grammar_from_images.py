import argparse
import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from treelib import Tree

from src.utils.grammars_utils.ascfg import Grammar, merge_grammars
from src.utils.grammars_utils.bayesian_merging import BayesianMerger
from src.utils.lattice_utils.data_manipulations import resize_facade, crop_facade
from src.utils.lattice_utils.split_lines import infer_split_lines
from src.utils.lattice_utils.lattice import Lattice
from src.utils.grammars_utils.general_grammar import Facade, parse_facade, get_tree_loss
from src.utils.dataset_metadata import SEGFACADEDataset


parser = argparse.ArgumentParser()
parser.add_argument("imgs_dir")
parser.add_argument("masks_dir")
parser.add_argument("output_dir")
parser.add_argument("--window_labels", nargs="*", type=int, required=True)
parser.add_argument("--output_pickle_name", default="grammar.pickle")
parser.add_argument("--checkpoints_dir", default="checkpoints")
parser.add_argument("--checkpoint_every_n", type=int, default=100)
parser.add_argument("--imgs_ext", default="png")
parser.add_argument("--downsampling_factor", default=0.17, type=float)
parser.add_argument("--background_label", default=0, type=int)
parser.add_argument("--n_random_draws", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=500)

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
    return img, mask[:, :, ::-1]


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


def get_best_parse_tree(lattice: Lattice) -> Tree:
    facade_nt = Facade(
        lattice=lattice,
        max_ground_floor=0.3,
        n_possible_splits=3,
        window_labels=args.window_labels,
    )
    trees = parse_facade(facade_nt, max_depth=-1)
    trees_losses = [get_tree_loss(tree) for tree in trees]
    return trees[np.argmin(trees_losses)]


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

    # create parse trees from lattices

    print("Creating parse trees of facades from lattices ...")
    general_parse_trees = [get_best_parse_tree(lattice) for lattice in tqdm(lattices)]

    # create ASCFG grammar from facades

    print("Creating instance-specific grammars ...")
    grammars = [
        Grammar.from_general_parse_tree(tree) for tree in tqdm(general_parse_trees)
    ]
    print("Merging into one stochastic grammar ...")
    grammar = merge_grammars(grammars)

    # induce generative grammar
    print("Training generative grammar ...")
    merger = BayesianMerger(
        w=2.0,
        strategy="random_draw",
        n_random_draws=args.n_random_draws,
    )
    grammar, loss_val = merger.perform_merging(
        grammar=grammar,
        lattices=lattices,
        n_epochs=args.n_epochs,
        checkpoint_dir=args.checkpoints_dir,
        checkpoint_every_n=args.checkpoint_every_n,
    )

    # save resulting grammar
    print(f"Training ended with loss {loss_val}")
    output_path = os.path.join(args.output_dir, args.output_pickle_name)
    print(f"Saving grammar to {output_path} ...")
    with open(output_path, "wb") as f:
        pickle.dump(grammar, f)
