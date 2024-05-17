import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from treelib import Tree

from src.utils.grammars_utils.ascfg import merge_grammars
from src.utils.grammars_utils.bayesian_merging import BayesianMerger
from src.utils.lattice_utils.data_manipulations import resize_facade, crop_facade
from src.utils.lattice_utils.split_lines import infer_split_lines
from src.utils.lattice_utils.lattice import Lattice
from src.utils.grammars_utils.general_grammar import Facade, parse_facade, get_tree_loss
from src.utils.dataset_metadata import SEGFACADEDataset


input_grammars_paths = [
    "checkpoints_1/checkpoint_150.pickle",
    "checkpoints_2/checkpoint_150.pickle",
    "checkpoints_3/checkpoint_150.pickle",
]

imgs_dirs = [
    "input_data/input_1/b",
    "input_data/input_2/b",
    "input_data/input_3/b",
]

masks_dirs = [
    "input_data/input_1/a",
    "input_data/input_2/a",
    "input_data/input_3/a",
]

window_labels = [2, 4]
background_label = 0
downsampling_factor = 0.17


def load_facade_and_mask(
    facade_name: str, imgs_dir, masks_dir
) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(os.path.join(imgs_dir, f"{facade_name}.png"))
    mask = cv2.imread(os.path.join(masks_dir, f"{facade_name}.png"))
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
        window_labels=window_labels,
    )
    trees = parse_facade(facade_nt, max_depth=-1)
    trees_losses = [get_tree_loss(tree) for tree in trees]
    return trees[np.argmin(trees_losses)]


if __name__ == "__main__":

    # load images and masks
    facades: list[tuple[np.ndarray, np.ndarray]] = []
    for imgs_dir, masks_dir in zip(imgs_dirs, masks_dirs):
        facades_names = [os.path.splitext(fname)[0] for fname in os.listdir(imgs_dir)]
        facades += [
            load_facade_and_mask(
                facade_name=facade_name, imgs_dir=imgs_dir, masks_dir=masks_dir
            )
            for facade_name in tqdm(facades_names)
        ]
    print(f"Loaded f{len(facades)} facades")

    # downsample facade's images and masks

    print("Downsampling images and masks ...")
    downsampled_facades: list[tuple[np.ndarray, np.ndarray]] = []
    for img, mask in tqdm(facades):
        downsampled_facades.append(
            resize_facade(img, mask, resizing_factor=downsampling_factor)
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
            background_label=background_label,
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
    for path in tqdm(input_grammars_paths):
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
