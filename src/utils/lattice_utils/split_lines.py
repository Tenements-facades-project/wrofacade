"""This module provide functions for finding split lines
on a facade image (and segmentation mask) that may be used
to build the rectangular lattice of the facade
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.cluster import DBSCAN


def suppress_lines(
    lines_inds: np.ndarray, dbscan_params: dict[str, Any] | None = None
) -> list[int]:
    """Given indexes of split lines with redundant lines,
    extracts a smaller representative set of lines with the DBSCAN
    clustering

    Args:
        lines_inds (1D array of ints): array of pixel indexes of lines
        dbscan_params (dict): dict of form {"param_name" -> "param_value"}
            for sklearn DBSCAN
            (see the code for default values)

    Returns:
        list: indexes of extracted lines set
    """
    dbscan_default_params = {"min_samples": 1, "eps": 20.0}
    dbscan_params = (
        {**dbscan_default_params, **dbscan_params}
        if dbscan_params
        else dbscan_default_params
    )
    dbscan = DBSCAN(**dbscan_params)
    clusters = dbscan.fit_predict(lines_inds.reshape((-1, 1)))
    return [
        int(np.mean(lines_inds[clusters == cluster])) for cluster in np.unique(clusters)
    ]


def infer_split_lines(
    mask: np.ndarray,
    dbscan_params: dict[str, Any] | None = None,
    min_change_coef: float = 0.05,
) -> tuple[list[int], list[int]]:
    """Extracts facade image split lines using pixels' classes
    labels

    Args:
        mask (2D array): pixelwise class labels mask
        dbscan_params (dict): DBSCAN params dict to be passed to the `suppress_lines`
            function
        min_change_coef (float): number of range [0-1],
            minimum part of pixels in a dimension that are changing along
            the line so that the line is assumed to be a split line

    Returns:
        tuple (hor_lines_inds, ver_lines_inds), where `hor_lines_inds` is a list
            of horizontal split lines indexes and `ver_lines_inds` is a list of
            vertical split lines indexes
    """
    height, width = mask.shape

    left_right_transitions = (mask[:, :-1] != mask[:, 1:]).astype("int")
    ver_lines_inds = np.argwhere(
        np.sum(left_right_transitions, axis=0) > int(height * min_change_coef)
    )
    ver_lines_inds = suppress_lines(ver_lines_inds, dbscan_params=dbscan_params)

    up_down_transitions = (mask[:-1, :] != mask[1:, :]).astype("int")
    hor_lines_inds = np.argwhere(
        np.sum(up_down_transitions, axis=1) > int(width * min_change_coef)
    )
    hor_lines_inds = suppress_lines(hor_lines_inds, dbscan_params=dbscan_params)

    for inds_list, dim_size in zip((ver_lines_inds, hor_lines_inds), (width, height)):
        if 0 in inds_list:
            inds_list.remove(0)
        if (dim_size - 1) in inds_list:
            inds_list.remove(dim_size - 1)

    return hor_lines_inds, ver_lines_inds
