"""This module provides general grammar tools,
i.e. tools to create a parse tree from a rectangular
facade's lattice
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal, Iterable
from itertools import product
import numpy as np
from treelib import Tree

from ..lattice_utils.lattice import ImgRange, Lattice


class GeneralTerminal(ImgRange):
    """Base class for terminal symbols in general grammar,
    i.e. symbols that do not undergo further division

    Attributes:
        name (str): name of the facade element represented
            by the terminal, or some string used to identify
            the symbol
    """

    def __init__(self, img: np.ndarray, mask: np.ndarray, name: str):
        self.name: str = name
        super().__init__(img=img, mask=mask)


class GeneralNonterminal(ABC, ImgRange):
    """Base class for non-terminal symbols in general grammar,
    i.e. symbols that can be further parsed into a set
    of other symbols

    Attributes:
        lattice: Lattice object of the image area represented
            by the nonterminal
        name (str): name of the facade part represented
            by the nonterminal, or some string used to identify
            the symbol
    """

    def __init__(self, lattice: Lattice, name: str):
        self.lattice = lattice
        self.name = name
        img, mask = self.lattice.assemble_lattice()
        super().__init__(img=img, mask=mask)

    @abstractmethod
    def possible_splits(self) -> list[tuple[GeneralNonterminal, ...] | GeneralTerminal]:
        """Obtains all possible parsings (splits into other symbols)
        of the non-terminal symbol

        Each possible parsing can be one of two forms:
        - split into a (ordered) set of non-terminals
        - transformation to the terminal symbol (lexical production)

        Returns:
            list of possible parsings, where each parsing is either
                a tuple of non-terminals (split) or a terminal (lexical
                production)

        """
        pass


# for type hints
GeneralSymbol = GeneralTerminal | GeneralNonterminal


class OtherNonterminal(GeneralNonterminal):
    """Universal general nonterminal (i.e. it can be applied
    to any lattice, no matter what facade part it represents)

    The split rule of this type of nonterminal is as follows:
    - if the lattice is 1x1 (one element in the lattice), then it produces
    one GeneralTerminal
    - if `split_direction` == `horizontal`, then it splits into the first
    lattice row in its lattice (the row most in the up) an into the rest of the
    lattice
    - if `split_direction` == `vertical`, then it splits into the first
    column from the left in the lattice and the rest of the lattice

    The `split_direction` of all symbol's child nonterminals is the opposite of the
    `split_direction` of the symbol

    """

    def __init__(
        self,
        lattice: Lattice,
        name: str,
        split_direction: Literal["horizontal", "vertical"],
    ):
        super().__init__(
            lattice=lattice,
            name=name,
        )
        if split_direction not in ("horizontal", "vertical"):
            raise ValueError(f"Invalid split_direction: {split_direction}")
        self.split_direction = split_direction

    def possible_splits(self) -> list[tuple[GeneralNonterminal, ...] | GeneralTerminal]:
        shape = self.lattice.ranges.shape

        if shape == (1, 1):
            return [
                GeneralTerminal(
                    img=self.img,
                    mask=self.mask,
                    name=self.name + "T",
                )
            ]

        new_names = [f"{self.name}{i}" for i in range(2)]
        if self.split_direction == "horizontal":
            if shape[0] == 1:
                self_copy = deepcopy(self)
                self_copy.split_direction = "vertical"
                return [(self_copy,)]
            first = OtherNonterminal(
                lattice=self.lattice[0, :],
                name=new_names[0],
                split_direction="vertical",
            )
            rest = OtherNonterminal(
                lattice=self.lattice[1:, :],
                name=new_names[1],
                split_direction="vertical",
            )
        else:
            if shape[1] == 1:
                self_copy = deepcopy(self)
                self_copy.split_direction = "horizontal"
                return [(self_copy,)]
            first = OtherNonterminal(
                lattice=self.lattice[:, 0],
                name=new_names[0],
                split_direction="horizontal",
            )
            rest = OtherNonterminal(
                lattice=self.lattice[:, 1:],
                name=new_names[1],
                split_direction="horizontal",
            )
        return [(first, rest)]


class Floor(GeneralNonterminal):
    """General nonterminal symbol representing one floor of a facade

    It splits along vertical split lines. The lines to split are chosen
    by checking which lattice columns contain tiles of class `window`
    (or some other that indicates there's a window), but the rule is designed
    so that no window is adjacent to more than one split line
    (i.e. the goal is: one window -> one segment)

    Args:
        window_labels: labels that have to be treated as window

    """

    def __init__(self, lattice: Lattice, window_labels: Iterable[int]):
        super().__init__(lattice=lattice, name="Floor")
        self.window_labels: Iterable[int] = window_labels

    def possible_splits(self) -> list[tuple[GeneralNonterminal, ...] | GeneralTerminal]:

        # find split lines that are adjacent to windows
        # but no window is adjacent to more than one split line
        split_inds = []
        flag = False
        for i, col in enumerate(self.lattice.ranges.transpose()):
            # get list of class labels present in the current column
            major_labels = [rng.most_frequent_label() for rng in col]
            if flag:
                # previous column contained window - don't consider this column
                # but if there's no windows in this column, reset flag
                if all((label not in major_labels) for label in self.window_labels):
                    flag = False
                continue
            if any((label in major_labels) for label in self.window_labels):
                # window in current column - add split line
                split_inds.append(i)
                flag = True

        nonterminals = []

        # add 0 and (number_of_columns) to split inds
        # (i.e. beginning of the first and end of the last)
        if 0 not in split_inds:
            split_inds = [0] + split_inds
        split_inds = split_inds + [len(self.lattice.ranges.transpose())]

        # add nonterminals to the list
        for left_ind, right_ind in zip(split_inds[:-1], split_inds[1:]):
            nonterminals.append(
                OtherNonterminal(
                    lattice=self.lattice[:, left_ind:right_ind],
                    name="FloorPart",
                    split_direction="horizontal",
                )
            )

        return [tuple(nonterminals)]


class GroundFloor(GeneralNonterminal):
    """General nonterminal symbol representing a ground floor of a facade

    It splits vertically by finding strongest split lines (see `Lattice.n_strongest_splits`
    method). When splitting, it returns (`max_n_splits` - `min_n_splits` + 1)
    possible splits, each split with a different number of split lines

    """

    def __init__(self, lattice: Lattice, min_n_splits: int, max_n_splits: int):
        super().__init__(lattice=lattice, name="GroundFloor")
        self.min_n_splits = min_n_splits
        self.max_n_splits = max_n_splits

    def possible_splits(self) -> list[tuple[GeneralNonterminal, ...] | GeneralTerminal]:

        # add possible split result for each number of splits in desired range
        possible_splits = []
        for n_splits in range(self.min_n_splits, self.max_n_splits + 1):

            # get `n_splits` strongest split lines
            split_inds = self.lattice.n_strongest_splits(
                n=n_splits, split_direction="vertical"
            )
            split_inds.sort()

            # add 0 and (number_of_columns) to split inds
            # (i.e. beginning of the first and end of the last)
            if 0 not in split_inds:
                split_inds = [0] + split_inds
            split_inds = split_inds + [len(self.lattice.ranges.transpose())]

            # obtain nonterminals and add possible split to the list
            nonterminals = []
            for left_ind, right_ind in zip(split_inds[:-1], split_inds[1:]):
                nonterminals.append(
                    OtherNonterminal(
                        lattice=self.lattice[:, left_ind:right_ind],
                        name="GroundFloorPart",
                        split_direction="horizontal",
                    )
                )
            possible_splits.append(tuple(nonterminals))

        return possible_splits


class Facade(GeneralNonterminal):
    """General nonterminal symbol representing an entire facade

    It splits into the following sequence of general nonterminals:
    (`OtherNonterminal`, `Floor` x n, `GroundFloor`)
    where n is the number of floors in the facade (EXCLUDING the ground floor).
    To split the ground floor out of the rest of the facade, `Lattice.n_strongest_splits`
    method is used. Floors are extracted by search for pixel classes that indicate
    window (the goal is that each floor contains exactly one row of windows).
    The lattice rows above the most high windows row states an OtherNonterminal instance

    Args:
        lattice: `Lattice` object to create `Facade` symbol from
        max_ground_floor (float [0;1]): the maximum relative part of facade height
            occupied with ground floor
        n_possible_splits: number of possible splits to be returned when splitting
            (each split has a different split line dividing ground floor from the rest
            of facade)
        window_labels: labels that have to be treated as window

    """

    def __init__(
        self,
        lattice: Lattice,
        max_ground_floor: float,
        n_possible_splits: int,
        window_labels: Iterable[int],
    ):
        super().__init__(lattice=lattice, name="Facade")
        self.max_ground_floor = max_ground_floor
        self.n_possible_splits = n_possible_splits
        self.window_labels: Iterable[int] = window_labels

    def possible_splits(self) -> list[tuple[GeneralNonterminal, ...] | GeneralTerminal]:

        # find split lines that are adjacent to windows
        # but no window is adjacent to more than one split line
        windows_split_inds = []
        flag = False
        for i, row in enumerate(self.lattice.ranges):
            # get list of class labels present in the current row
            major_labels = [rng.most_frequent_label() for rng in row]
            if flag:
                # previous row contained window - don't consider this row
                # but if there's no windows in this row, reset flag
                if all((label not in major_labels) for label in self.window_labels):
                    flag = False
                continue
            if any((label in major_labels) for label in self.window_labels):
                # window in current row - add split line
                windows_split_inds.append(i)
                flag = True

        # extract possible floors - ground floor splits
        rows_lens = [row[0].img.shape[0] for row in self.lattice.ranges]
        cum_row_parts = np.cumsum(rows_lens) / np.sum(rows_lens)
        floors_offset = np.argmin(np.abs(cum_row_parts - (1 - self.max_ground_floor)))
        possible_gf_split_inds = self.lattice[floors_offset:, :].n_strongest_splits(
            n=self.n_possible_splits, split_direction="horizontal"
        )
        possible_gf_split_inds = [ind + floors_offset for ind in possible_gf_split_inds]

        possible_splits = []
        # add possible split for each ground floor possible split ind
        for gf_split_ind in possible_gf_split_inds:
            nonterminals = []
            rel_windows_split_inds = [
                ind for ind in windows_split_inds if ind < gf_split_ind
            ]
            if not rel_windows_split_inds:
                # ground floor is too high - no windows detected above
                continue
            if rel_windows_split_inds[0] != 0:
                nonterminals.append(
                    OtherNonterminal(
                        lattice=self.lattice[: rel_windows_split_inds[0], :],
                        name="UpPart",
                        split_direction="vertical",
                    )
                )
            rel_windows_split_inds = rel_windows_split_inds + [gf_split_ind]
            for up_ind, down_ind in zip(
                rel_windows_split_inds[:-1], rel_windows_split_inds[1:]
            ):
                nonterminals.append(
                    Floor(
                        lattice=self.lattice[up_ind:down_ind, :],
                        window_labels=self.window_labels,
                    )
                )
            nonterminals.append(
                GroundFloor(
                    lattice=self.lattice[gf_split_ind:, :],
                    min_n_splits=3,
                    max_n_splits=7,
                )
            )
            possible_splits.append(tuple(nonterminals))

        return possible_splits


def split_leaves(tree: Tree) -> list[Tree]:
    """Given a parse tree of a facade with general symbols,
    extracts all possible splits from the tree's leaves
    that are nonterminals and creates new trees, each tree
    representing a possible split case
    (i.e. nonterminals leaves are not leaves anymore as they
    were split and have children symbols).
    The list of all possible trees is returned

    Args:
        tree: input tree with leaves to be split

    Returns:
        all possible trees with leaves after split

    """
    leaves = [
        leaf for leaf in tree.leaves() if not isinstance(leaf.data, GeneralTerminal)
    ]
    leaves_possible_splits = [leaf.data.possible_splits() for leaf in leaves]
    new_trees = []
    for leaves_splits in product(*leaves_possible_splits):
        new_tree = Tree(tree=tree)
        for leaf, leaf_split in zip(leaves, leaves_splits):
            if isinstance(leaf_split, GeneralTerminal):
                new_tree.create_node(
                    f"{leaf.identifier}T",
                    f"{leaf.identifier}T",
                    parent=leaf.identifier,
                    data=leaf_split,
                )
            else:
                for i, leaf_child in enumerate(leaf_split):
                    new_tree.create_node(
                        f"{leaf.identifier}-{i}",
                        f"{leaf.identifier}-{i}",
                        parent=leaf.identifier,
                        data=leaf_child,
                    )
        new_trees.append(new_tree)
    return new_trees


def is_tree_complete(tree: Tree) -> bool:
    """Returns True if the tree is complete, i.e.
    there is no leaf that is nonterminal and could
    be split
    """
    return not any(
        [isinstance(leaf.data, GeneralNonterminal) for leaf in tree.leaves()]
    )


def parse_facade(
    facade_nt: Facade,
    max_depth: int,
) -> list[Tree]:
    """Splits a `Facade` general nonterminal symbol
    and creates parse trees

    Args:
        facade_nt: `Facade` general nonterminal class instance
        max_depth: maximum number of split levels;
            if -1, then split until complete

    Returns:
        list of all possible parse trees extracted

    Notes:
        As `max_depth` is imposed, some nonterminal leaves may
        remain in parse trees in the result

    """
    tree_root = Tree()
    tree_root.create_node("facade", "0", data=facade_nt)
    trees_new = [tree_root]
    trees_old = None
    if max_depth == -1:
        while any([not is_tree_complete(tree) for tree in trees_new]):
            trees_old = trees_new
            trees_new = sum([split_leaves(tree) for tree in trees_old], start=[])
    else:
        for _ in range(max_depth):
            trees_old = trees_new
            trees_new = sum([split_leaves(tree) for tree in trees_old], start=[])
    return trees_new


def symbol_name_loss(symbols: list[GeneralNonterminal]) -> float:
    """Computes loss value for a list of general nonterminal
    symbols that are assumed to have similar dimensions
    (e.g. facade floors)

    The loss value is the sum of two terms: variance of symbols heights
    and the variance of symbols widths
    """
    heights = [symbol.img.shape[0] for symbol in symbols]
    widths = [symbol.img.shape[1] for symbol in symbols]
    return np.var(heights) + np.var(widths)


def get_symbols_dict(tree: Tree) -> dict[str, list[GeneralNonterminal]]:
    """Creates dictionary from the tree of general symbols,
    where keys are symbols names and values are lists of nonterminals with the name
    that occurred in the tree
    """
    symbols_dict = {}
    for symbol in (
        node.data
        for node in tree.all_nodes_itr()
        if isinstance(node.data, GeneralNonterminal)
    ):
        name = symbol.name
        if name in symbols_dict:
            symbols_dict[name].append(symbol)
        else:
            symbols_dict[name] = [symbol]
    return symbols_dict


def get_tree_loss(tree: Tree) -> float:
    """Get loss value for the tree

    The loss is calculated using variances of dimensions
    of symbols with the same name (as they are assumed
    to have similar dimensions)
    """
    symbols_dict = get_symbols_dict(tree)
    return np.nanmean(
        [symbol_name_loss(symbols_list) for symbols_list in symbols_dict.values()]
    )
