from __future__ import annotations

from copy import deepcopy
from typing import Literal, Callable
import uuid
from uuid import UUID
import numpy as np
import pandas as pd
import cv2
from treelib import Node, Tree
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    validator,
    NonNegativeInt,
    root_validator,
)
import networkx as nx

from ..lattice_utils.lattice import ImgRange
from .general_grammar import GeneralSymbol, GeneralTerminal, GeneralNonterminal


class Terminal:
    """Class representing terminal symbol in the grammar,
    i.e. some facade image area that is not transformed
    further

    Attributes:
        img_range (ImgRange): ImgRange object with an image area associated
            with the terminal symbol
        most_frequent_label (int): most frequent pixel label in the mask,
            calculated automatically when creating the object

    """

    def __init__(self, img_range: ImgRange):
        self.img_range: ImgRange = img_range
        self.most_frequent_label: int = self.img_range.most_frequent_label()

    def resize(self, height: int, width: int) -> ImgRange:
        """Returns the image range (image with mask) associated
        with the terminal resized to desired dimensions

        Args:
            height (int): desired output height (pixels)
            width: (int): desired output width (pixels)

        Returns:
            resized ImgRange object
        """
        output_shape = (width, height)
        resized_img = cv2.resize(self.img_range.img, output_shape)
        resized_mask = cv2.resize(
            self.img_range.mask, output_shape, interpolation=cv2.INTER_NEAREST
        )
        return ImgRange(img=resized_img, mask=resized_mask)


class Nonterminal:
    """Class representing non-terminal symbol in the grammar,
    i.e. an element that is not a final symbol and should
    be further processed by a production rule

    Attributes:
        reachable_labels (set[int]): all pixels' classes labels that
            can be reached by applying production rules on this
            symbol

    """

    def __init__(self, reachable_labels: set[int]):
        self.reachable_labels: set[int] = reachable_labels


# for type hints
Symbol = Terminal | Nonterminal


class ImgBox(BaseModel):
    """Represents a rectangular area of a facade image with
    an associated grammar symbol

    The area can be retrieved from the image `facade_image` as follows:
    >>> facade_img[box.up:box.down, box.left:box.right]
    """

    left: NonNegativeInt
    right: NonNegativeInt
    up: NonNegativeInt
    down: NonNegativeInt
    is_terminal: bool
    symbol_id: UUID

    @root_validator(pre=True)
    def check_bounds_consistency(cls, values):
        if values.get("up") >= values.get("down"):
            raise ValueError("`up` should be lower than `down`")
        if values.get("left") >= values.get("right"):
            raise ValueError("`left` should be lower than `right`")
        return values


class RuleAttribute(BaseModel):
    """Represents an attribute of a production rule, contains
    information about relative sizes of RHS symbols

    Attributes:
        sizes(list): list of float values from range (0.0; 1.0),
            ith value is the size of ith RHS symbol divided
            by the length of the LHS symbol; values must sum up to 1.0

    """

    sizes: list[NonNegativeFloat]

    @validator("sizes")
    def sizes_sum_to_one(cls, sizes: list[NonNegativeFloat]):
        if np.abs(np.sum(sizes) - 1.0) > 1e-5:
            raise ValueError("`sizes` must sum up to 1.0")
        return sizes


class ProductionRule:
    """Represents split grammar production rule, which transforms
    an input nonterminal symbol (called LHS symbol) into a list of output
    (RHS) symbols

    They are two types of production rules:
    - split production - splits the input nonterminal symbol into a list of nonterminal symbols
    that correspond to parts of the LHS area (i.e. it splits some symbol into "smaller" symbols)
    - lexical production - transforms the input nonterminal symbol into a terminal symbol

    Attributes:
        split_direction ({"horizontal", "vertical"} | None): informs whether a rule performs
            a horizontal or a vertical split
        rule_type ({"split", "lexical"}): type of the rule
        lhs (UUID): ID of the LHS nonterminal symbol in some external nonterminals dict
        rhs (tuple[UUID, ...] | UUID): IDs of the output nonterminal symbols (for `rule_type="split"`)
            or ID of the output terminal symbol (for `rule_type="lexical"`)
        attributes (list[RuleAttribute] | None): list of rule attributes, each attributes is
            a set of relative RHS sizes; during a production, one attribute is selected
            (randomly or manually by the user)
        attributes_probs (list[float]): list of attributes probabilities used when choosing
            the attribute by random; the ith value is the probability of the ith attribute
            in the `self.attributes`

    """

    def __init__(
        self,
        split_direction: Literal["horizontal", "vertical"] | None,
        rule_type: Literal["split", "lexical"],
        lhs: UUID,
        rhs: tuple[UUID, ...] | UUID,
        attributes: list[RuleAttribute] | None,
        attributes_probs: list[float] | None,
    ):
        self.check_rule_type_rhs_consistency(rule_type=rule_type, rhs=rhs)
        self.split_direction: Literal["horizontal", "vertical"] | None = split_direction
        self.rule_type: Literal["split", "lexical"] = rule_type
        self.lhs: UUID = lhs
        self.rhs: tuple[UUID, ...] | UUID = rhs
        self.attributes: list[RuleAttribute] | None = attributes
        self.attributes_probs: list[float] | None = attributes_probs

    @staticmethod
    def check_rule_type_rhs_consistency(
        rule_type: Literal["split", "lexical"], rhs: tuple[UUID, ...] | UUID
    ) -> None:
        if rule_type == "split":
            if not isinstance(rhs, tuple):
                raise ValueError(
                    "For rule type `split` the `rhs` value should be a tuple of UUID"
                )
        elif rule_type == "lexical":
            if not isinstance(rhs, UUID):
                raise ValueError(
                    "For rule type `lexical` the `rhs` value should be a UUID object"
                )
        else:
            raise ValueError(f"Invalid rule type: {rule_type}")

    @staticmethod
    def __divide_area(
        lower_bound: int, higher_bound: int, attribute: RuleAttribute
    ) -> list[tuple[int, int]]:
        """Returns indices of division, given an area bounds
        and an attribute with output relative sizes
        """
        all_size = higher_bound - lower_bound
        if all_size < 2:
            raise ValueError("`upper_bound - lower_bound` should be at least 2")
        segments_bounds: list[tuple[int, int]] = []
        offset_prev = None
        offset_next = lower_bound
        for segment_size in attribute.sizes[:-1]:
            offset_prev = offset_next
            offset_next = offset_prev + round(all_size * segment_size)
            segments_bounds.append((offset_prev, offset_next))
        segments_bounds.append((offset_next, higher_bound))
        return segments_bounds

    @staticmethod
    def __clean_area_division(
        segments_bounds: list[tuple[int, int]], rhs: tuple[UUID, ...]
    ) -> tuple[list[tuple[int, int]], tuple[UUID, ...]]:
        """Given area division, obtains division that does not
        contain parts of size lower than 2

        Returns:
            tuple: (cleaned_segments, cleaned_segments_ids)
        """
        segments_lengths = [
            segment_end - segment_start
            for segment_start, segment_end in segments_bounds
        ]
        ids_to_remove = [
            i for i, segment_len in enumerate(segments_lengths) if segment_len < 2
        ]
        if not ids_to_remove:
            return segments_bounds, rhs
        if len(ids_to_remove) == len(segments_lengths):
            return ([(segments_bounds[0][0], segments_bounds[-1][1])], (rhs[0],))
        new_bounds, new_rhs = [], []
        cur_lower_bound = 0
        for i, (segment, segment_id) in enumerate(zip(segments_bounds[:-1], rhs[:-1])):
            if i not in ids_to_remove:
                lower_bound, higher_bound = segment
                new_bounds.append((cur_lower_bound, higher_bound))
                new_rhs.append(segment_id)
                cur_lower_bound = higher_bound
        if (len(segments_bounds) - 1) in ids_to_remove:
            new_bounds[-1] = (new_bounds[-1][0],) + (segments_bounds[-1][1],)
        else:
            new_bounds.append((cur_lower_bound, segments_bounds[-1][1]))
            new_rhs.append(rhs[-1])
        return new_bounds, tuple(new_rhs)

    def choose_attribute_idx(self) -> int:
        """Choose randomly index i,
        i in [0, len(self.attributes))
        """
        rng = np.random.default_rng()
        return rng.choice(len(self.attributes), p=self.attributes_probs)

    def __call__(self, box: ImgBox, attribute_idx: int | None = None) -> list[ImgBox]:
        """Performs production using the rule on an input box

        Args:
            box (ImgBox): input box
            attribute_idx (int | None): the index of the attribute in the `self.attributes` to be used
                during production; if None, then the attribute is chosen randomly

        Returns:
            list[ImgBox]: list of boxes corresponding to subsequent split outputs
        """
        if self.rule_type == "lexical":
            return [
                ImgBox(
                    left=box.left,
                    right=box.right,
                    up=box.up,
                    down=box.down,
                    is_terminal=True,
                    symbol_id=self.rhs,
                )
            ]
        boxes: list[ImgBox] = []
        if not attribute_idx:
            attribute_idx = self.choose_attribute_idx()
        if self.split_direction == "horizontal":
            new_segments, new_rhs = self.__clean_area_division(
                self.__divide_area(
                    lower_bound=box.up,
                    higher_bound=box.down,
                    attribute=self.attributes[attribute_idx],
                ),
                self.rhs,
            )
            for (segment_up, segment_down), symbol_id in zip(new_segments, new_rhs):
                boxes.append(
                    ImgBox(
                        left=box.left,
                        right=box.right,
                        up=segment_up,
                        down=segment_down,
                        is_terminal=False,
                        symbol_id=symbol_id,
                    )
                )
        else:
            new_segments, new_rhs = self.__clean_area_division(
                self.__divide_area(
                    lower_bound=box.left,
                    higher_bound=box.right,
                    attribute=self.attributes[attribute_idx],
                ),
                self.rhs,
            )
            for (segment_left, segment_right), symbol_id in zip(new_segments, new_rhs):
                boxes.append(
                    ImgBox(
                        left=segment_left,
                        right=segment_right,
                        up=box.up,
                        down=box.down,
                        is_terminal=False,
                        symbol_id=symbol_id,
                    )
                )
        return boxes

    @classmethod
    def infer_rule(
        cls,
        lhs_box: ImgBox,
        rhs_general_symbols: list[GeneralSymbol],
        rhs_ids: list[UUID],
        is_lexical: bool,
    ) -> ProductionRule:
        """Given an input box and a list of output GeneralSymbols
        instances (obtained with a general grammar), constructs
        a production rule such that it produces from the lhs box
        a boxes list with dimensions of output general symbols

        Args:
            lhs_box (ImgBox): LHS box
            rhs_general_symbols: list[GeneralSymbol]: list of general symbols
                with shapes that should be produced from the LHS box
            rhs_ids (list[UUID]): list of IDs that rule will assign to its output
                symbols
            is_lexical (bool): whether a lexical rule is to be inferred

        Returns:
            ProductionRule: output production rule object
        """
        lhs_height = lhs_box.down - lhs_box.up
        lhs_width = lhs_box.right - lhs_box.left
        if is_lexical:
            rule_type = "lexical"
            split_direction = None
            attributes = None
            attributes_probs = None
            rhs_ids = rhs_ids[0]
        else:
            rule_type = "split"
            attributes_probs = [1.0]
            rhs_ids = tuple(rhs_ids)
            if rhs_general_symbols[0].mask.shape[0] == lhs_height:
                split_direction = "vertical"
                attributes = [
                    RuleAttribute(
                        sizes=[
                            nt.mask.shape[1] / lhs_width for nt in rhs_general_symbols
                        ]
                    )
                ]
            elif rhs_general_symbols[0].mask.shape[1] == lhs_width:
                split_direction = "horizontal"
                attributes = [
                    RuleAttribute(
                        sizes=[
                            nt.mask.shape[0] / lhs_height for nt in rhs_general_symbols
                        ]
                    )
                ]
            else:
                raise ValueError

        return cls(
            split_direction=split_direction,
            lhs=lhs_box.symbol_id,
            rhs=rhs_ids,
            rule_type=rule_type,
            attributes=attributes,
            attributes_probs=attributes_probs,
        )

    def get_closest_attribute(self, sizes: list[float]) -> tuple[RuleAttribute, int]:
        """Finds attribute of the rule with values closest to
        given

        Args:
            sizes: given sizes
                (relative sizes of rhs shapes, must sum up to 1.0)

        Returns:
            RuleAttribute: attribute of the rule that is closest
                to given sizes
            int: index of the closest attribute in `self.attributes`

        """
        elementwise_diffs = np.vstack(
            [attr.sizes for attr in self.attributes]
        ) - np.array(sizes)
        distances = np.sum(np.power(elementwise_diffs, 2), axis=1)
        min_ind = np.argmin(distances)
        return self.attributes[min_ind], min_ind


class StartProduction:
    """Represents a production that may be called as the first step
    in the grammar inference (generation) process; it produces the starting
    shape (the first nonterminal) which is the root of the parse tree

    Args:
        start_shape: tuple[int, int]: the shape of the produced box
        start_nonterminal_id (UUID): ID of the nonterminal in some external
            nonterminals dict that is to be associated with the produced box;
            it will be the nonterminal in the root of the parse tree
    """

    def __init__(self, start_shape: tuple[int, int], start_nonterminal_id: UUID):
        self.split_direction = "vertical"  # by convention
        self.lhs = "START"
        self.rhs = start_nonterminal_id
        self.rule_type: Literal["start"] = "start"
        self.start_shape: tuple[int, int] = start_shape

    def __call__(self) -> ImgBox:
        return ImgBox(
            up=0,
            down=self.start_shape[0],
            left=0,
            right=self.start_shape[1],
            symbol_id=self.rhs,
            is_terminal=False,
        )


def general_symbol_to_symbol(gen_symbol: GeneralSymbol) -> Symbol:
    """Converts a general symbol obtained with general grammar
    to an ASCFG symbol
    """
    if isinstance(gen_symbol, GeneralTerminal):
        return Terminal(ImgRange(gen_symbol.img, gen_symbol.mask))
    if isinstance(gen_symbol, GeneralNonterminal):
        _, mask = gen_symbol.lattice.label_major_vote().assemble_lattice()
        return Nonterminal(reachable_labels=set(np.unique(mask)))
    raise ValueError(f"Unknown general symbol type {type(gen_symbol)}")


class ParseNode(Node):
    """Represents a node in a facade parse tree obtained
    from a grammar

    Attributes:
        is_terminal (bool): whether the node is a terminal node
        symbol (Symbol): a Symbol object (Terminal or Nonterminal) associated
            with the node
        box (ImgBox): an ImgBox object associated with the node, contains information
            about the location of the `self.symbol` on the facade image
        rule (ProductionRule | None): a production rule that' been used to split the node's
            symbol into other symbols (the node's children nodes); None if the node is
            a leaf in the parse tree
        attribute_idx (int): the index in the `self.rule.attributes` of the attribute chosen to
            produce node's children; None if the node is a leaf in the parse tree
        rule_prob (float): probability of the `self.rule` in the grammar that produced
            the parse tree; None if the node is a leaf in the parse tree

    """

    def __init__(
        self,
        symbol: Symbol | None,
        box: ImgBox | None,
        rule: ProductionRule | None,
        attribute_idx: int | None,
        rule_prob: float | None,
    ):
        super().__init__()
        self.is_terminal: bool = isinstance(symbol, Terminal)
        self.symbol: Symbol = symbol
        self.box: ImgBox = box
        self.rule: ProductionRule | None = rule
        self.attribute_idx: int | None = attribute_idx
        self.rule_prob: float | None = rule_prob


class ParseTree(Tree):
    """Represents a parse tree of a facade image obtained with a grammar"""

    def __init__(self, start_production: StartProduction, start_production_prob: float):
        super().__init__(node_class=ParseNode)
        self.start_production: StartProduction = start_production
        self.start_production_probe: float = start_production_prob

    def logprob(self) -> float:
        """Returns log-probability of the tree"""
        logprob = np.log(self.start_production_probe)
        for parse_node in self.all_nodes_itr():
            if not parse_node.is_terminal:
                logprob += np.log(parse_node.rule_prob)
        return logprob

    def assemble_image(self) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the facade image and its class labels mask
        using parse tree leaves (build up the image from the terminals
        from the tree)

        Returns:
            tuple: (image, mask)
        """
        img_height, img_width = self.start_production.start_shape
        img = np.empty((img_height, img_width, 3), dtype=np.uint8)
        mask = np.empty((img_height, img_width), dtype=np.uint8)
        for parse_node in self.all_nodes_itr():
            if parse_node.is_terminal:
                box: ImgBox = parse_node.box
                terminal_range = parse_node.symbol.resize(
                    height=box.down - box.up, width=box.right - box.left
                )
                img[box.up : box.down, box.left : box.right] = terminal_range.img
                mask[box.up : box.down, box.left : box.right] = terminal_range.mask
        return img, mask


class Grammar:
    """Represents a two-dimensional attributed stochastic
    context-free grammar (2D-ASCFG) as defined in the paper
    """

    def __init__(
        self,
        nonterminals: dict[UUID, Nonterminal],
        terminals: dict[UUID, Terminal],
        rules: list[ProductionRule | StartProduction],
    ):
        self.nonterminals: dict[UUID, Nonterminal] = nonterminals
        self.terminals: dict[UUID, terminals] = terminals
        self.rules_df: pd.DataFrame = self.__obtain_rules_df(rules)
        self.rules_graph: nx.DiGraph | None = None

    def __obtain_rules_df(
        self, rules: list[ProductionRule | StartProduction]
    ) -> pd.DataFrame:
        rules = pd.Series(rules)
        lhs = rules.apply(lambda x: x.lhs)
        rhs = rules.apply(lambda x: x.rhs)
        split_direction = rules.apply(lambda x: x.split_direction)
        rule_type = rules.apply(lambda x: x.rule_type)
        lhs_probs = 1 / lhs.value_counts()
        rule_prob = lhs.apply(lambda x: lhs_probs.loc[x])
        return pd.DataFrame(
            {
                "rules": rules,
                "lhs": lhs,
                "rhs": rhs,
                "split_direction": split_direction,
                "rule_type": rule_type,
                "rule_prob": rule_prob,
            }
        )

    @classmethod
    def from_general_parse_tree(cls, tree: Tree) -> Grammar:
        # queue (in form of a list) of tuples:
        # (parent_node_identifier, parent_node_box
        symbols_queue: list[tuple[str, ImgBox]] = []

        # get root id and box, add to the queue
        root_node_id = "0"
        root_node = tree.get_node(root_node_id)
        start_rule = StartProduction(
            start_shape=root_node.data.mask.shape,
            start_nonterminal_id=uuid.uuid4(),
        )
        root_box = start_rule()
        symbols_queue.append((root_node_id, root_box))

        # variables to store the result
        rules: list[ProductionRule | StartProduction] = [start_rule]
        terminals: dict[UUID, Terminal] = {}
        nonterminals: dict[UUID, Nonterminal] = {
            root_box.symbol_id: general_symbol_to_symbol(root_node.data)
        }

        # process the queue until it's empty
        while symbols_queue:
            parent_identifier, parent_box = symbols_queue.pop(0)
            children = tree.children(parent_identifier)

            if len(children) == 1 and isinstance(children[0].data, GeneralTerminal):
                # lexical production
                child = children[0]
                child_id = uuid.uuid4()
                child_gen_symbol = child.data
                rule = ProductionRule.infer_rule(
                    lhs_box=parent_box,
                    rhs_general_symbols=[child_gen_symbol],
                    rhs_ids=[child_id],
                    is_lexical=True,
                )
                terminals[child_id] = general_symbol_to_symbol(child_gen_symbol)
                rules.append(rule)

            elif (
                len(children) == 1 and isinstance(children[0].data, GeneralNonterminal)
            ) or len(children) > 1:
                # split production
                children_ids = [uuid.uuid4() for child in children]
                children_gen_symbols = [child.data for child in children]
                for child_id, child_gen_symbol in zip(
                    children_ids, children_gen_symbols
                ):
                    assert child_id not in nonterminals
                    nonterminals[child_id] = general_symbol_to_symbol(child_gen_symbol)
                rule = ProductionRule.infer_rule(
                    lhs_box=parent_box,
                    rhs_general_symbols=children_gen_symbols,
                    rhs_ids=children_ids,
                    is_lexical=False,
                )
                children_boxes = rule(parent_box, attribute_idx=0)
                rules.append(rule)
                for child, child_box in zip(children, children_boxes):
                    symbols_queue.append((child.identifier, child_box))

            else:
                # a nonterminal leaf - additional lexical production should be added
                terminal_id = uuid.uuid4()
                parent_gen_symbol = tree.get_node(parent_identifier).data
                terminal = Terminal(
                    ImgRange(img=parent_gen_symbol.img, mask=parent_gen_symbol.mask)
                )
                terminals[terminal_id] = terminal
                rules.append(
                    ProductionRule(
                        split_direction=None,
                        rule_type="lexical",
                        lhs=parent_box.symbol_id,
                        rhs=terminal_id,
                        attributes=None,
                        attributes_probs=None,
                    )
                )

        # create Grammar object
        return Grammar(nonterminals=nonterminals, terminals=terminals, rules=rules)

    def get_all_rules_for_lhs(
        self,
        lhs: UUID | str,
        rule_chooser: Callable[[ProductionRule], bool] = lambda x: True,
    ) -> tuple[list[ProductionRule | StartProduction] | None, list[float] | None]:
        """Given ID of a nonterminal ('lhs'),
        returns a subset of all rules from the grammar that are applicable
        to this nonterminal (i.e. their LHS is `lhs`)
        along with their probabilities

        Args:
            lhs: UUID of LHS nonterminal
            rule_chooser: Callable, getting ProductionRule object as an argument
                and returning `True` if the rule is accepted and `False` otherwise;
                all rules are accepted by default

        Returns:
            tuple: (rules_for_lhs, rules_probs_for_lhs),
                where rules_for_lhs is a list of all accepted rules with desired LHS
                nad rules_probs_for_lhs are probabilities of these rules,
                normalized so that they sum up to 1.0
                if there's no rule satisfying condition, returns (`None`, `None`)

        """
        rules_df_for_lhs = self.rules_df[self.rules_df["lhs"] == lhs]
        rules_df_for_lhs = rules_df_for_lhs[
            rules_df_for_lhs["rules"].apply(rule_chooser)
        ]
        if rules_df_for_lhs.shape[0] == 0:
            # no rules satisfying condition
            return None, None
        rules_for_lhs = rules_df_for_lhs["rules"].tolist()
        rules_probs_for_lhs = rules_df_for_lhs["rule_prob"]
        rules_probs_for_lhs = rules_probs_for_lhs / rules_probs_for_lhs.sum()
        rules_probs_for_lhs = rules_probs_for_lhs.tolist()
        return rules_for_lhs, rules_probs_for_lhs

    def get_rule_for_lhs(
        self,
        lhs: UUID | str,
        rule_chooser: Callable[[ProductionRule], bool] = lambda x: True,
    ) -> tuple[ProductionRule | StartProduction | None, float | None]:
        """Chooses a production rule from the grammar
        that is applicable to a nonterminal;
        the rule is chosen randomly, according to
        rules probabilities
        """
        rules_for_lhs, rules_probs_for_lhs = self.get_all_rules_for_lhs(
            lhs, rule_chooser=rule_chooser
        )
        if not rules_for_lhs:
            return None, None
        rng = np.random.default_rng()
        idx = rng.choice(len(rules_for_lhs), p=rules_probs_for_lhs)
        return rules_for_lhs[idx], rules_probs_for_lhs[idx]

    def generate_parse_tree(
        self,
        rules_choosers: dict[int, Callable[[ProductionRule], bool]] | None = None,
        rebuild_rules_graph: bool = False,
    ) -> ParseTree:
        """Performs production of the grammar

        Start shape is generated with one of the starting rules that are
        in the grammar. Next, the shape is split with grammar rules
        and shapes are transformed until they are become terminal.
        The parse tree is returned, which can be later use to obtain
        generated lattice and generated facade's mask and image

        Returns:
            ParseTree: result parse tree object
        """
        if rules_choosers is None:
            rules_choosers = dict()

        if any(level > 1 for level in rules_choosers.keys()):
            raise ValueError("Levels greater than 1 are currently not supported")

        # build rules graph if necessary
        if self.rules_graph is None or rebuild_rules_graph:
            self.__build_rules_graph()

        # get start production

        # get all start productions that satisfy all rule choosers
        if not rules_choosers:
            start_rule, start_rule_prob = self.get_rule_for_lhs("START")
            rules_choosers = dict()
        else:
            rel_start_prods_for_levels = []
            for chooser_level, chooser_fun in rules_choosers.items():
                relevant_start_productions_tuple = self.__relevant_start_productions(
                    chooser_fun=chooser_fun, chooser_level=chooser_level
                )
                if relevant_start_productions_tuple is None:
                    raise RuntimeError("No parse tree satisfying rule chooser")
                rel_start_prods_for_levels.append(relevant_start_productions_tuple[0])
            rel_start_prods = (
                rel_start_prods_for_levels[0]
                if len(rel_start_prods_for_levels) == 1
                else list(
                    set(rel_start_prods_for_levels[0]).intersection(
                        *rel_start_prods_for_levels[1:]
                    )
                )
            )
            if not rel_start_prods:
                raise RuntimeError("No parse tree satisfying all rule choosers")

            # choose start production randomly
            rel_start_prods_probs = [
                self.rules_graph.nodes[rule]["rule_prob"] for rule in rel_start_prods
            ]
            probs_sum = sum(rel_start_prods_probs)
            rel_start_prods_probs = [prob / probs_sum for prob in rel_start_prods_probs]
            chosen_ind = np.random.choice(
                len(rel_start_prods_probs), p=rel_start_prods_probs
            )
            start_rule, start_rule_prob = (
                rel_start_prods[chosen_ind],
                rel_start_prods_probs[chosen_ind],
            )

        # create ParseTree object
        parse_tree = ParseTree(
            start_production=start_rule, start_production_prob=start_rule_prob
        )

        # get the start box and split recursively
        # queue of tuples: (box, node_identifier)
        boxes_queue: list[tuple[ImgBox, str | UUID | None]] = [(start_rule(), None)]
        while boxes_queue:
            box, parent_id = boxes_queue.pop(0)

            if box.is_terminal:
                # get the symbol from grammar
                symbol = self.terminals[box.symbol_id]
                rule, rule_prob, rule_attribute_idx = None, None, None

            else:
                # get the level in tree in which node will be created
                level = (
                    parse_tree.level(parent_id) if parent_id is not None else -1
                ) + 2

                # get the symbol from the grammar
                symbol = self.nonterminals[box.symbol_id]

                # get the rule from the grammar and its attribute
                rule, rule_prob = self.get_rule_for_lhs(
                    box.symbol_id,
                    rule_chooser=rules_choosers.get(level, lambda x: True),
                )
                if rule is None:
                    raise RuntimeError
                rule_attribute_idx = (
                    rule.choose_attribute_idx() if rule.rule_type != "lexical" else None
                )

            # add node to the parse tree
            node = ParseNode(
                symbol=symbol,
                box=box,
                rule=rule,
                attribute_idx=rule_attribute_idx,
                rule_prob=rule_prob,
            )
            parse_tree.add_node(node, parent=parent_id)

            # if not terminal, add child boxes to the queue
            if not box.is_terminal:
                child_boxes = rule(box=box, attribute_idx=rule_attribute_idx)
                for child_box in child_boxes:
                    boxes_queue.append((child_box, node.identifier))

        return parse_tree

    def merge_nonterminals(
        self, nonterm_1_id: UUID, nonterm_2_id: UUID, new_nonterm_id: UUID
    ) -> Grammar:
        """Return new grammar with two given nonterminals
        merged into a new nonterminal

        Args:
            nonterm_1_id: ID of the first nonterminal to merge in the
                nonterminals dict
            nonterm_2_id: ID of the second nonterminal to merge in the
                nonterminals dict
            new_nonterm_id: ID of the new nonterminal

        Returns:
            Grammar: new grammar object with merged nonterminals into
                one new nonterminal
        """

        reachable_labels = self.nonterminals[nonterm_1_id].reachable_labels
        if reachable_labels != self.nonterminals[nonterm_2_id].reachable_labels:
            raise ValueError("Nonterminals not label-compatible")
        rules = deepcopy(self.rules_df.rules.tolist())
        nonterminals = self.nonterminals.copy()
        for nt_id in (nonterm_1_id, nonterm_2_id):
            nonterminals.pop(nt_id)
            for rule in rules:
                if rule.lhs == nt_id:
                    rule.lhs = new_nonterm_id
                if rule.rule_type == "split":
                    if nt_id in rule.rhs:
                        rule.rhs = tuple(
                            new_nonterm_id if symbol_id == nt_id else symbol_id
                            for symbol_id in rule.rhs
                        )
                else:
                    if nt_id == rule.rhs:
                        rule.rhs = new_nonterm_id
        new_nonterm = Nonterminal(reachable_labels=reachable_labels)
        nonterminals[new_nonterm_id] = new_nonterm
        return Grammar(terminals=self.terminals, nonterminals=nonterminals, rules=rules)

    def __build_rules_graph(self) -> None:
        """Builds graph of grammar's production rules

        In the graph, nodes correspond to rules. An edge from node i to node j
        exists if when in grammar production rule i is applied on a terminal N
        and children nonterminals are obtained: N_1, N_2, ..., N_k,
        it is possible that the rule j will be applied on one of childs
        (i.e. rule j is applicable for at least one of nonterminals N_1, N_2, ..., N_k)
        """
        rules_graph = nx.DiGraph()

        # create graph's nodes
        for index, row in self.rules_df.iterrows():
            rules_graph.add_node(
                row.loc["rules"],
                rule_prob=row.loc["rule_prob"],
                is_start=(row.loc["rule_type"] == "start"),
            )

        # create graph's edges
        for index, row in self.rules_df.iterrows():
            if row.loc["rule_type"] == "lexical":
                continue
            for rhs_id in (
                row.loc["rhs"]
                if (row.loc["rule_type"] != "start")
                else (row.loc["rhs"],)
            ):
                applicable_rules, applicable_rules_probs = self.get_all_rules_for_lhs(
                    rhs_id
                )
                for rule, rule_prob in zip(applicable_rules, applicable_rules_probs):
                    rules_graph.add_edge(row.loc["rules"], rule)

        self.rules_graph = rules_graph

    def __relevant_start_productions(
        self, chooser_fun: Callable[[ProductionRule], bool], chooser_level: int
    ) -> tuple[list[StartProduction], list[float]] | None:
        if self.rules_graph is None:
            raise AssertionError
        rel_start_rules, rel_start_rules_probs = [], []
        for start_rule, start_rule_prob in zip(*self.get_all_rules_for_lhs("START")):
            k_distant_nodes = nx.descendants_at_distance(
                self.rules_graph, start_rule, chooser_level
            )
            if any(chooser_fun(rule) for rule in k_distant_nodes):
                rel_start_rules.append(start_rule)
                rel_start_rules_probs.append(start_rule_prob)
        if not rel_start_rules:
            return None
        return rel_start_rules, rel_start_rules_probs


def merge_grammars(grammars: list[Grammar]) -> Grammar:
    """Given a list of grammars, creates a new merged grammar
    that contains all grammars

    The result grammar's nonterminals dict, terminals dict and
    rules list contain nonterminals, terminals and rules from
    all grammars, respectively (if two terminals/nonterminals
    among terminals/nonterminals from all grammars have the same
    ID, ValueError is raised). The result grammars contains
    all start rules, so the production of the result grammar
    is just random choice of the start rule and then production
    with the one of the input grammars which start rule
    has been chosen

    Args:
        grammars: list of Grammar objects to be merged into
            one grammar

    Returns:
        Grammar: merged grammar object
    """
    rules: list[ProductionRule | StartProduction] = (
        grammars[0].rules_df["rules"].tolist()
    )
    nonterminals: dict[UUID, Nonterminal] = grammars[0].nonterminals
    terminals: dict[UUID, Terminal] = grammars[0].terminals

    for grammar in grammars[1:]:
        nonterminals_overlap = set(nonterminals.keys()).intersection(
            grammar.nonterminals.keys()
        )
        terminals_overlap = set(terminals.keys()).intersection(grammar.terminals.keys())
        if len(nonterminals_overlap) != 0:
            raise ValueError(
                f"Overlapping nonterminals keys found: {nonterminals_overlap}"
            )
        if len(terminals_overlap) != 0:
            raise ValueError(f"Overlapping terminals keys found: {terminals_overlap}")

        rules += grammar.rules_df["rules"].tolist()
        nonterminals = {**nonterminals, **grammar.nonterminals}
        terminals = {**terminals, **grammar.terminals}

    return Grammar(nonterminals=nonterminals, terminals=terminals, rules=rules)
