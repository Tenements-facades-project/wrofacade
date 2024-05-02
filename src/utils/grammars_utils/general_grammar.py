from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal, Iterable
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from treelib import Tree

from ..lattice_utils.lattice import ImgRange, Lattice


class GeneralSymbol(ABC, ImgRange):
    """Base class for all general grammar symbols
    (i.e. elements of facades' lattices' parse trees)

    Attributes:
        name (str): name of the facade element represented
            by the symbol, or some string used to identify
            the symbol
        label (int): an integer label assigned to the symbol and
            used in the mask
        mask_original (2D array): original pixels labels mask fragment
            corresponding to the range of the input corresponding to
            the symbol
    """

    def __init__(self, img: np.ndarray, mask: np.ndarray, name: str, label: int):
        self.name: str = name
        self.label: int = label
        self.mask_original = mask
        super().__init__(img=img, mask=None)


class GeneralTerminal(GeneralSymbol):
    """Base class for terminal symbols in general grammar,
    i.e. symbols that do not undergo further division

    """

    def __init__(self, img: np.ndarray, mask: np.ndarray, name: str, label: int):
        super().__init__(img=img, mask=mask, name=name, label=label)


class GeneralNonterminal(GeneralSymbol):
    """Base class for non-terminal symbols in general grammar,
    i.e. symbols that can be further parsed into a set
    of other symbols
    """

    def __init__(self, lattice: Lattice, name: str, label: int):
        self.lattice = lattice
        img, mask = self.lattice.assemble_lattice()
        super().__init__(img=img, mask=mask, name=name, label=label)

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
        serial_number: int,
        split_direction: Literal["horizontal", "vertical"],
    ):
        super().__init__(
            lattice=lattice, name=f"other_{serial_number}", label=serial_number
        )
        if split_direction not in ("horizontal", "vertical"):
            raise ValueError(f"Invalid split_direction: {split_direction}")
        self.split_direction = split_direction
        self.serial_number = serial_number

    def possible_splits(self) -> list[tuple[GeneralNonterminal, ...] | GeneralTerminal]:
        shape = self.lattice.ranges.shape

        if shape == (1, 1):
            return [
                GeneralTerminal(
                    img=self.img,
                    mask=self.mask_original,
                    name=self.name,
                    label=self.label,
                )
            ]

        new_serial_numbers = [int(f"{self.serial_number}{i}") for i in range(2)]
        if self.split_direction == "horizontal":
            if shape[0] == 1:
                self_copy = deepcopy(self)
                self_copy.split_direction = "vertical"
                return [(self_copy,)]
            first = OtherNonterminal(
                lattice=self.lattice[0, :],
                serial_number=new_serial_numbers[0],
                split_direction="vertical",
            )
            rest = OtherNonterminal(
                lattice=self.lattice[1:, :],
                serial_number=new_serial_numbers[1],
                split_direction="vertical",
            )
        else:
            if shape[1] == 1:
                self_copy = deepcopy(self)
                self_copy.split_direction = "horizontal"
                return [(self_copy,)]
            first = OtherNonterminal(
                lattice=self.lattice[:, 0],
                serial_number=new_serial_numbers[0],
                split_direction="horizontal",
            )
            rest = OtherNonterminal(
                lattice=self.lattice[:, 1:],
                serial_number=new_serial_numbers[1],
                split_direction="horizontal",
            )
        return [(first, rest)]


class Floor(GeneralNonterminal):
    """General nonterminal symbol representing one floor of a facade

    It splits along vertical split lines. The lines to split are chosen
    by checking which lattice columns contain tiles of class `window`
    (or some other that indicate there's a window), but the rule is designed
    so that no window is adjacent to more than one split line
    (i.e. the goal is: one window -> one segment)

    Args:
        window_labels: labels that have to be treated as window

    """

    def __init__(self, lattice: Lattice, window_labels: Iterable[int] = (8, 3)):
        super().__init__(lattice=lattice, name="Floor", label=10)
        self.window_labels: Iterable[int] = window_labels

    def possible_splits(self) -> list[tuple[GeneralNonterminal, ...] | GeneralTerminal]:
        lattice_mv = self.lattice.label_major_vote()
        split_inds = []
        flag = False
        for i, col in enumerate(lattice_mv.ranges.transpose()):
            major_labels = [rng.most_freqent_label() for rng in col]
            if flag:
                if all((label not in major_labels) for label in self.window_labels):
                    flag = False
                continue
            if any((label in major_labels) for label in self.window_labels):
                split_inds.append(i)
                flag = True
        nonterminals = []
        if 0 not in split_inds:
            split_inds = [0] + split_inds
        split_inds = split_inds + [len(lattice_mv.ranges.transpose())]
        for left_ind, right_ind in zip(split_inds[:-1], split_inds[1:]):
            nonterminals.append(
                OtherNonterminal(
                    lattice=self.lattice[:, left_ind:right_ind],
                    serial_number=100 + left_ind,
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
        super().__init__(lattice=lattice, name="GroundFloor", label=9)
        self.min_n_splits = min_n_splits
        self.max_n_splits = max_n_splits

    def possible_splits(self) -> list[tuple[GeneralNonterminal, ...] | GeneralTerminal]:
        possible_splits = []
        for n_splits in range(self.min_n_splits, self.max_n_splits + 1):
            split_inds = self.lattice.n_strongest_splits(
                n=n_splits, split_direction="vertical"
            )
            split_inds.sort()
            if 0 not in split_inds:
                split_inds = [0] + split_inds
            split_inds = split_inds + [len(self.lattice.ranges.transpose())]
            nonterminals = []
            for left_ind, right_ind in zip(split_inds[:-1], split_inds[1:]):
                nonterminals.append(
                    OtherNonterminal(
                        lattice=self.lattice[:, left_ind:right_ind],
                        serial_number=100 + left_ind,
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
        self, lattice: Lattice, max_ground_floor: float, n_possible_splits: int,
        window_labels: Iterable[int] = (8, 3)
    ):
        super().__init__(lattice=lattice, name="Facade", label=0)
        self.max_ground_floor = max_ground_floor
        self.n_possible_splits = n_possible_splits
        self.window_labels: Iterable[int] = window_labels

    def possible_splits(self) -> list[tuple[GeneralNonterminal, ...] | GeneralTerminal]:
        # extract split indexes at windows
        windows_split_inds = []
        flag = False
        for i, row in enumerate(self.lattice.ranges):
            major_labels = [rng.most_freqent_label() for rng in row]
            if flag:
                if all((label not in major_labels) for label in self.window_labels):
                    flag = False
                continue
            if any((label in major_labels) for label in self.window_labels):
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
        for gf_split_ind in possible_gf_split_inds:
            nonterminals = []
            rel_windows_split_inds = [
                ind for ind in windows_split_inds if ind < gf_split_ind
            ]
            if rel_windows_split_inds[0] != 0:
                nonterminals.append(
                    OtherNonterminal(
                        lattice=self.lattice[: rel_windows_split_inds[0], :],
                        serial_number=50,
                        split_direction="vertical",
                    )
                )
            rel_windows_split_inds = rel_windows_split_inds + [gf_split_ind]
            for up_ind, down_ind in zip(
                rel_windows_split_inds[:-1], rel_windows_split_inds[1:]
            ):
                nonterminals.append(
                    Floor(
                        lattice=self.lattice[up_ind:down_ind, :], window_labels=self.window_labels
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
    representing a possible set of nonterminals split
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


def add_symbol_lines(
    parse_img,
    parse_tree,
    left_bound,
    right_bound,
    up_bound,
    down_bound,
    split_direction,
    line_width=3,
) -> np.ndarray:
    """Function aimed at visualization; creates a copy
    of given facade image and adds bounding boxes of symbols
    from the parse tree recursively

    """
    parse_img = parse_img.copy()

    box_interior = parse_img[
        up_bound + line_width : down_bound - line_width,
        left_bound + line_width : right_bound - line_width,
    ].copy()
    parse_img[up_bound:down_bound, left_bound:right_bound] = [0, 0, 255]
    parse_img[
        up_bound + line_width : down_bound - line_width,
        left_bound + line_width : right_bound - line_width,
    ] = box_interior

    start_ind = 0
    if split_direction == "horizontal":
        for root_child in parse_tree.children(parse_tree.root):
            height = root_child.data.img.shape[0]
            parse_img = add_symbol_lines(
                parse_img,
                parse_tree.subtree(root_child.identifier),
                left_bound,
                right_bound,
                up_bound + start_ind,
                down_bound + start_ind + height,
                "vertical",
            )
            start_ind += height
    else:
        for root_child in parse_tree.children(parse_tree.root):
            width = root_child.data.img.shape[1]
            parse_img = add_symbol_lines(
                parse_img,
                parse_tree.subtree(root_child.identifier),
                left_bound + start_ind,
                right_bound + start_ind + width,
                up_bound,
                down_bound,
                split_direction="horizontal",
            )
            start_ind += width
    return parse_img


def visualize_tree_symbols(tree):
    """Visualizes a general symbols
    parse tree on the image of the facade
    """
    parse_img = tree.get_node("0").data.img
    parse_img = add_symbol_lines(
        parse_img,
        tree,
        0,
        parse_img.shape[1] - 1,
        0,
        parse_img.shape[0] - 1,
        "horizontal",
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(parse_img[:, :, ::-1])
    plt.show()


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
    where keys are symbols names and values are list of nonterminals with the name
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
    return np.mean(
        [symbol_name_loss(symbols_list) for symbols_list in symbols_dict.values()]
    )
