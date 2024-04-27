from __future__ import annotations

from typing import Literal
import numpy as np

from ...visualize.img_mask import draw_facade_with_split_lines


class ImgRange:
    """Represents a rectangular area
    on the facade image
    (along with its class labels)

    Attributes:
        img (3D array): part of the image
        mask (2D array): class labels of pixels from img
    """

    def __init__(self, img: np.ndarray, mask: np.ndarray):
        self.img: np.ndarray = img
        self.mask: np.ndarray = mask

    def most_frequent_label(self) -> int:
        """Returns pixel class label that occurs in the
        `self.mask` most frequently
        """
        labels, counts = np.unique(self.mask, return_counts=True)
        return labels[np.argmax(counts)]

    def label_major_vote(self) -> np.ndarray:
        """Returns array of the same shape as `self.mask`
        but with all pixels with the same class label.
        The class label is the most frequent label in
        `self.mask`
        """
        return np.full(self.mask.shape, fill_value=self.most_frequent_label())


class Lattice:
    """Represents an irregular rectangular lattice
    that is fit to some facade image

    Attributes:
        ranges (2D array of `ImgRange` objects): array of ranges which
            represents the lattice; the position of a range corresponds
            to the position of area represented by the range on
            the image
    """

    def __init__(self, ranges: np.ndarray):
        if not isinstance(ranges, np.ndarray) or ranges.ndim != 2:
            raise ValueError("`ranges` should be a 2-dimensional array")
        self.ranges: np.ndarray = ranges

    def __getitem__(self, ind):
        """Enables user to index the lattice like a simple
        array, for example if `l` is a Lattice object with
        5x5 ranges array, `l[2:4, 1:]` will return new Lattice object
        with the range array being the l.range[2:4, 1:]

        Args:
            ind: indices or slices to index `ranges` array

        Returns:
            new Lattice object

        Notes:
            the method takes care of the dimensionality of the `ranges` array
            in the new Lattice object (i.e. so that it's always a 2D array)
        """
        if isinstance(ind, int):
            return Lattice(self.ranges[ind].reshape((1, -1)))
        if isinstance(ind, tuple):
            ind_1, ind_2 = ind
            if isinstance(ind_1, int) and isinstance(ind_2, int):
                return Lattice(np.array([[self.ranges[ind]]]))
            if isinstance(ind_1, slice) and isinstance(ind_2, slice):
                return Lattice(self.ranges[ind])
            if isinstance(ind_1, slice) and isinstance(ind_2, int):
                return Lattice(self.ranges[ind].reshape((-1, 1)))
            if isinstance(ind_1, int) and isinstance(ind_2, slice):
                return Lattice(self.ranges[ind].reshape((1, -1)))
        return Lattice(self.ranges[ind])

    @classmethod
    def from_lines(
        cls,
        img: np.ndarray,
        mask: np.ndarray,
        vertical_lines_inds: np.ndarray | list,
        horizontal_lines_inds: np.ndarray | list,
    ):
        """Creates `Lattice` object from a facade image, its
        segmentation mask and indices of horizontal and vertical
        split lines that form the lattice

        Args:
            img (3D array): image of a facade
            mask (2D array): segmentation mask for `img`
            vertical_lines_inds: indexes of pixels columns on the image that
                are vertical split lines
            horizontal_lines_inds: indexes of pixels rows on the image that
                are horizontal split lines

        Returns:
            `Lattice` class object
        """
        height, width = mask.shape
        rows_bounds = np.concatenate(([0], horizontal_lines_inds, [height]))
        cols_bounds = np.concatenate(([0], vertical_lines_inds, [width]))
        ranges = [
            [
                ImgRange(
                    img=img[up_bound:down_bound, left_bound:right_bound],
                    mask=mask[up_bound:down_bound, left_bound:right_bound],
                )
                for left_bound, right_bound in zip(cols_bounds[:-1], cols_bounds[1:])
            ]
            for up_bound, down_bound in zip(rows_bounds[:-1], rows_bounds[1:])
        ]
        return cls(ranges=np.array(ranges))

    def assemble_lattice(self) -> tuple[np.ndarray, np.ndarray]:
        """Builds image and mask from
        the lattice

        Returns:
            tuple: (facade_img, facade_mask)
        """
        rows = [
            np.concatenate([img_rng.img for img_rng in row], axis=1)
            for row in self.ranges
        ]
        res_img = np.concatenate(rows, axis=0)

        rows = [
            np.concatenate([img_rng.mask for img_rng in row], axis=1)
            for row in self.ranges
        ]
        res_mask = np.concatenate(rows, axis=0)

        return res_img, res_mask

    def plot(self) -> None:
        """Plots the entire facade image and segmentation mask
        built from the lattice
        """
        ver_lines_inds = np.cumsum([r.img.shape[1] for r in self.ranges[0]])[:-1]
        hor_lines_inds = np.cumsum([r.img.shape[0] for r in self.ranges[:, 0]])[:-1]

        img, mask = self.assemble_lattice()

        draw_facade_with_split_lines(
            img=img,
            mask=mask,
            hor_lines_inds=hor_lines_inds,
            ver_lines_inds=ver_lines_inds,
        )

    def label_major_vote(self) -> Lattice:
        """Returns new lattice in which ranges masks
        are entirely filled with their corresponding
        most frequent label
        """
        new_ranges = [
            [ImgRange(img=rng.img, mask=rng.label_major_vote()) for rng in row]
            for row in self.ranges
        ]
        return Lattice(ranges=np.array(new_ranges))

    def n_strongest_splits(
        self, n: int, split_direction: Literal["horizontal", "vertical"]
    ) -> list[int]:
        """Returns indexes of lattice splits (split lines)
        that divide the lattice most efficiently

        For each tile in the lattice, the major vote lattice is obtained
        and then number of differences between pixels classes on both sides
        of the line is calculated. For example, if `split_direction`=`"horizontal"`,
        then the first index of the result `ind_1` used as follows: `up_part = lattice[:ind_1, :]`,
        `down_part = lattice[ind_1:, :]` gives the best division.
        """
        major_vote_lattice = self.label_major_vote()
        if split_direction == "horizontal":
            m_rows = major_vote_lattice.ranges.shape[0]
            rows_classes_diffs = []
            prev_row = major_vote_lattice[0, :].assemble_lattice()[1]
            for next_ind in range(1, m_rows):
                next_row = major_vote_lattice[next_ind, :].assemble_lattice()[1]
                rows_classes_diffs.append(np.sum(prev_row[-1, :] != next_row[0, :]))
                prev_row = next_row
            return (np.flip(np.argsort(rows_classes_diffs))[:n] + 1).tolist()
        m_cols = major_vote_lattice.ranges.shape[1]
        cols_classes_diffs = []
        prev_col = major_vote_lattice[:, 0].assemble_lattice()[1]
        for next_ind in range(1, m_cols):
            next_col = major_vote_lattice[:, next_ind].assemble_lattice()[1]
            cols_classes_diffs.append(np.sum(prev_col[:, -1] != next_col[:, 0]))
            prev_col = next_col
        return (np.flip(np.argsort(cols_classes_diffs))[:n] + 1).tolist()
