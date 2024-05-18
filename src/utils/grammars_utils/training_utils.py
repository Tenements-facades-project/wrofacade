import os
import numpy as np
import cv2
from treelib import Tree

from ..lattice_utils.lattice import Lattice
from ..lattice_utils.split_lines import infer_split_lines
from .general_grammar import Facade, parse_facade, get_tree_loss


def load_facade_and_mask(
    facade_name: str, imgs_dir, masks_dir
) -> tuple[np.ndarray, np.ndarray]:
    """Loads image and segmentation mask of a facade

    Args:
        facade_name: name of the facade file without extension
        imgs_dir: path to the directory with images
        masks_dir: path to the directory with masks

    Returns:
        tuple: img, mask,
            where img is facade's image in form 3D numpy array (BGR)
            and mask is facade's segmentation mask in form of 3D array (RGB)
    """
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
    """Infers split lines for a facade image and
    returns Lattice object
    """
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


def get_best_parse_tree(lattice: Lattice, window_labels: list[int]) -> Tree:
    """Finds best parse tree for a facade's lattice,
    using general grammar symbols
    """
    facade_nt = Facade(
        lattice=lattice,
        max_ground_floor=0.3,
        n_possible_splits=3,
        window_labels=window_labels,
    )
    trees = parse_facade(facade_nt, max_depth=-1)
    trees_losses = [get_tree_loss(tree) for tree in trees]
    return trees[np.argmin(trees_losses)]
