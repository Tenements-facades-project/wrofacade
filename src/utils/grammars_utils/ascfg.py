from __future__ import annotations

from copy import deepcopy
from typing import Literal
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
            for (segment_up, segment_down), symbol_id in zip(
                self.__divide_area(
                    lower_bound=box.up,
                    higher_bound=box.down,
                    attribute=self.attributes[attribute_idx],
                ),
                self.rhs,
            ):
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
            for (segment_left, segment_right), symbol_id in zip(
                self.__divide_area(
                    lower_bound=box.left,
                    higher_bound=box.right,
                    attribute=self.attributes[attribute_idx],
                ),
                self.rhs,
            ):
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
        self, lhs: UUID | str
    ) -> tuple[list[ProductionRule | StartProduction], list[float]]:
        """Given ID of a nonterminal ('lhs'),
        returns all rules from the grammar that are applicable
        to this nonterminal (i.e. their LHS is `lhs`)
        along with their probabilities
        """
        rules_df_for_lhs = self.rules_df[self.rules_df["lhs"] == lhs]
        rules_for_lhs = rules_df_for_lhs["rules"].tolist()
        rules_probs_for_lhs = rules_df_for_lhs["rule_prob"].tolist()
        return rules_for_lhs, rules_probs_for_lhs

    def get_rule_for_lhs(
        self, lhs: UUID | str
    ) -> tuple[ProductionRule | StartProduction, float]:
        """Chooses a production rule from the grammar
        that is applicable to a nonterminal;
        the rule is chosen randomly, according to
        rules probabilities
        """
        rules_for_lhs, rules_probs_for_lhs = self.get_all_rules_for_lhs(lhs)
        rng = np.random.default_rng()
        idx = rng.choice(len(rules_for_lhs), p=rules_probs_for_lhs)
        return rules_for_lhs[idx], rules_probs_for_lhs[idx]

    def generate_parse_tree(self) -> ParseTree:
        """Performs production of the grammar

        Start shape is generated with one of the starting rules that are
        in the grammar. Next, the shape is split with grammar rules
        and shapes are transformed until they are become terminal.
        The parse tree is returned, which can be later use to obtain
        generated lattice and generated facade's mask and image

        Returns:
            ParseTree: result parse tree object
        """
        # get start production and create ParseTree object
        start_rule, start_rule_prob = self.get_rule_for_lhs("START")
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
                # get the symbol from the grammar
                symbol = self.nonterminals[box.symbol_id]

                # get the rule from the grammar and its attribute
                rule, rule_prob = self.get_rule_for_lhs(box.symbol_id)
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
