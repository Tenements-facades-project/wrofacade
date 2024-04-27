"""This module provides tools to visualize facade img
and its segmentation mask side by side
"""

import numpy as np
import matplotlib.pyplot as plt


def draw_facade_and_mask(
    img: np.ndarray,
    mask: np.ndarray,
    show_axis: bool = True,
) -> None:
    """Creates a two-axes plot of a facade (image and pixels classes mask)

    Args:
        img (3D array): image of the facade in BGR
        mask (2D array): pixels labels of the facade's image
        show_axis (bool): whether to retain X, Y axis on axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    numrows, numcols = mask.shape
    extent = (0.0, numcols, numrows, 0.0)

    axes[0].imshow(img[:, :, ::-1], extent=extent)
    axes[1].imshow(mask, extent=extent)

    if not show_axis:
        for ax in axes:
            ax.set_axis_off()

    plt.show()


def draw_facade_with_split_lines(
    img: np.ndarray,
    mask: np.ndarray,
    ver_lines_inds: list[int],
    hor_lines_inds: list[int],
    show_axis: bool = True,
) -> None:
    """Creates a two-axes plot of a facade (image and pixels classes mask)
    with vertical and horizontal split lines

    Args:
        img (3D array): image of the facade in BGR
        mask (2D array): pixels labels of the facade's image
        ver_lines_inds (list of ints): a list of vertical split lines indexes
        hor_lines_inds (list of ints): a list of horizontal split lines indexes
        show_axis (bool): whether to retain X, Y axis on axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    numrows, numcols = mask.shape
    extent = (0.0, numcols, numrows, 0.0)

    axes[0].imshow(img[:, :, ::-1], extent=extent)
    axes[1].imshow(mask, extent=extent)

    for ax in axes:
        for ind in hor_lines_inds:
            ax.axhline(y=ind, xmin=0, xmax=1, c="m")
        for ind in ver_lines_inds:
            ax.axvline(x=ind, ymin=0, ymax=1, c="r")
        if not show_axis:
            ax.set_axis_off()

    plt.show()
