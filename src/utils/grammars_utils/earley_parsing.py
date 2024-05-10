"""Implementation of the Earley Parser for 2D grammars
based on:

Martinovic A et al.
Early Parsing for 2D Stochastic
Context Free Grammars; Technical Report

"""
from __future__ import annotations

from itertools import product
from uuid import UUID
from warnings import warn
from pydantic import BaseModel, ConfigDict, NonNegativeInt, NonNegativeFloat
import numpy as np
import pandas as pd

from ..lattice_utils.lattice import Lattice
from .ascfg import (
    ProductionRule,
    StartProduction,
    Grammar,
    ImgBox,
    ParseTree,
    ParseNode,
)


class EarleyState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rule: ProductionRule | StartProduction
    rule_prob: NonNegativeFloat
    dot_position: NonNegativeInt = 0
    origin_position: tuple[int, int]
    bbox_x: NonNegativeInt
    bbox_y: NonNegativeInt
    bbox_X: NonNegativeInt
    bbox_Y: NonNegativeInt
    scanning_history: np.ndarray
    child_states: list[EarleyState] = []

    def __repr__(self):
        return f"""
        Earley State
        (i,j) = {self.origin_position}

        rule: {self.rule.lhs} -> {self.rule.rhs}
        {self.rule.rule_type} {self.rule.split_direction}

        bbox_x: {self.bbox_x}\tbbox_y: {self.bbox_y}
        bbox_X: {self.bbox_X}\tbbox_Y: {self.bbox_Y}

        {len(self.child_states)} child states
        """

    def is_complete(self) -> bool:
        rhs_length = (
            1 if self.rule.rule_type in ("lexical", "start") else len(self.rule.rhs)
        )
        return self.dot_position == rhs_length


def get_starting_states_from_history(
        states_history: dict[tuple[int, int], list[EarleyState]]
) -> list[EarleyState]:
    starting_states: list[EarleyState] = []
    for grid_pos in states_history:
        for state in states_history[grid_pos]:
            if state.rule.rule_type == "start" and state.is_complete():
                starting_states.append(state)
    return starting_states


def get_state_pixels_shape(state: EarleyState, lattice: Lattice) -> tuple[int, int]:
    state_lattice = lattice[state.bbox_y: state.bbox_Y, state.bbox_x: state.bbox_X]
    return state_lattice.assemble_lattice()[1].shape


class EarleyParser2D:
    def __init__(self, grammar: Grammar):
        self.grammar: Grammar = grammar

    def parse_lattice(
            self, lattice: Lattice, max_iterations: int = 5000
    ) -> tuple[list[ParseTree], dict[tuple[int, int], list[EarleyState]]]:
        # create states history dict and states queue
        lat_n_rows, lat_n_cols = lattice.ranges.shape
        states_history = {
            # in Earley chart first dim is col index
            grid_position: []
            for grid_position in product(range(lat_n_cols + 1), range(lat_n_rows + 1))
        }
        states_queue: list[EarleyState] = []

        # create starting state for each possible starting rule
        # and push them to the queue
        start_rules, start_rules_probs = self.grammar.get_all_rules_for_lhs("START")
        for start_rule, rule_prob in zip(start_rules, start_rules_probs):
            start_state = EarleyState(
                rule=start_rule,
                rule_prob=rule_prob,
                dot_position=0,
                origin_position=(0, 0),
                bbox_x=0,
                bbox_y=0,
                bbox_X=lat_n_cols,  # in Earley chart first dim is col index
                bbox_Y=lat_n_rows,
                scanning_history=np.full(
                    (lat_n_rows, lat_n_cols),  # in history, first dim is row index
                    np.nan,
                    dtype=object,
                ),
            )
            states_history[start_state.origin_position].append(start_state)
            states_queue.append(start_state)

        # process states until queue is empty
        # add each state to the states history
        i = -1
        while states_queue:
            i += 1
            if i == max_iterations:
                # TODO: why it happens?
                warn("Maximum number of iterations reached, parsing stopped!")
                break
            state = states_queue.pop(0)

            if state.is_complete():
                # state is complete (dot at the end)
                new_states = self.__complete(
                    complete_state=state, states_history=states_history
                )

            else:
                if state.rule.rule_type == "lexical":
                    scan_result_state = self.__scan(state=state, lattice=lattice)
                    if scan_result_state:
                        new_states = [scan_result_state]
                    else:
                        new_states = []
                else:
                    new_states = self.__predict(state=state)

            # if any new states, add to history and push to queue
            for new_state in new_states:
                states_history[new_state.origin_position].append(new_state)
                states_queue.append(new_state)

        # build parse trees for parsing paths
        parse_trees = self.__parse_trees_from_states_history(
            states_history=states_history, lattice=lattice
        )

        return parse_trees, states_history

    def __predict(
            self,
            state: EarleyState,
    ) -> list[EarleyState]:
        new_states: list[EarleyState] = []

        # extract the symbol to the right of the dot
        # or get start symbol id if StartProduction
        if isinstance(state.rule, StartProduction):
            symbol_id = state.rule.rhs
        else:
            symbol_id = state.rule.rhs[state.dot_position]

        # all possible expansions of the symbol
        rules, rules_probs = self.grammar.get_all_rules_for_lhs(symbol_id)
        for rule, rule_prob in zip(rules, rules_probs):
            new_states.append(
                EarleyState(
                    rule=rule,
                    rule_prob=rule_prob,
                    dot_position=0,
                    origin_position=state.origin_position,
                    bbox_x=state.origin_position[0],
                    bbox_y=state.origin_position[1],
                    bbox_X=state.bbox_X,
                    bbox_Y=state.bbox_Y,
                    scanning_history=state.scanning_history,
                )
            )
        return new_states

    def __scan(
            self,
            state: EarleyState,
            lattice: Lattice,
    ) -> EarleyState | None:

        # state is assumed to contain lexical production
        if state.rule.rule_type != "lexical":
            raise AssertionError("Only states with lexical production can be scanned")

        # get terminal most frequent label and check if it's consistent with lattice
        terminal = self.grammar.terminals[state.rule.rhs]
        terminal_label = terminal.img_range.most_freqent_label()
        i, j = state.origin_position
        try:
            cell_label = lattice.ranges[j, i].most_freqent_label()
        except IndexError:
            # TODO: find the error's cause
            warn("IndexError: a state outside lattice was produced")
            return None
        if terminal_label != cell_label:
            # state does not match the input lattice
            return None

        if not pd.isnull(state.scanning_history[j, i]):
            # scanned position in the grid has been already scanned
            return None

        updated_history = state.scanning_history.copy()
        updated_history[j, i] = state.rule.rhs

        return EarleyState(
            rule=state.rule,
            rule_prob=state.rule_prob,
            dot_position=1,
            origin_position=(i + 1, j),
            bbox_x=i,
            bbox_y=j,
            bbox_X=i + 1,
            bbox_Y=j + 1,
            scanning_history=updated_history,
        )

    def __check_complete_candidate(
            self, complete_state: EarleyState, candidate_state: EarleyState
    ) -> bool:
        # check if candidate is not complete
        if candidate_state.is_complete():
            return False

        # check if it's not a lexical production
        if candidate_state.rule.rule_type == "lexical":
            return False

        # check if candidate_state has matching symbol after the dot
        candidate_after_dot = (
            candidate_state.rule.rhs
            if candidate_state.rule.rule_type == "start"
            else candidate_state.rule.rhs[candidate_state.dot_position]
        )
        if complete_state.rule.lhs != candidate_after_dot:
            return False

        # check if candidate_state's bbox encompasses complete_state's bbox
        if any(
                [
                    complete_state.bbox_x < candidate_state.bbox_x,
                    complete_state.bbox_y < candidate_state.bbox_y,
                    complete_state.bbox_X > candidate_state.bbox_X,
                    complete_state.bbox_Y > candidate_state.bbox_Y,
                ]
        ):
            return False

        # check "scanning history - compatibility"
        H_complete = complete_state.scanning_history
        H_candidate = candidate_state.scanning_history
        if not np.all(
                (H_complete == H_candidate)
                | (pd.isnull(H_complete))
                | (pd.isnull(H_candidate))
        ):
            return False

        # check dimensions alingment
        if (
                complete_state.rule.split_direction == "vertical"
                and candidate_state.rule.split_direction == "vertical"
        ):
            if (
                    complete_state.bbox_Y != candidate_state.bbox_Y
            ) and candidate_state.dot_position != 0:
                return False
        if (
                complete_state.rule.split_direction == "horizontal"
                and candidate_state.rule.split_direction == "vertical"
        ):
            if (
                    complete_state.bbox_Y != candidate_state.bbox_Y
            ) and candidate_state.dot_position != 0:
                return False
        if (
                complete_state.rule.split_direction == "vertical"
                and candidate_state.rule.split_direction == "horizontal"
        ):
            if (
                    complete_state.bbox_X != candidate_state.bbox_X
            ) and candidate_state.dot_position != 0:
                return False
        if (
                complete_state.rule.split_direction == "horizontal"
                and candidate_state.rule.split_direction == "horizontal"
        ):
            if (
                    complete_state.bbox_X != candidate_state.bbox_X
            ) and candidate_state.dot_position != 0:
                return False
        return True

    def __complete(
            self,
            complete_state: EarleyState,
            states_history: dict[tuple[int, int], list[EarleyState]],
    ) -> list[EarleyState]:
        # extract all states that can be completed
        candidate_states = [
            candidate_state
            for candidate_state in states_history[
                complete_state.bbox_x, complete_state.bbox_y
            ]
            if self.__check_complete_candidate(
                complete_state=complete_state, candidate_state=candidate_state
            )
        ]

        # create states
        new_states = []
        p: ProductionRule = complete_state.rule

        for candidate_state in candidate_states:
            p_prim: ProductionRule = candidate_state.rule
            is_mu_null = (
                True
                if candidate_state.rule.rule_type == "start"
                else (candidate_state.dot_position + 1) == len(p_prim.rhs)
            )
            is_rule_vertical = (
                lambda x: x.rule_type == "lexical" or x.split_direction == "vertical"
            )

            if is_rule_vertical(p) and is_rule_vertical(p_prim):
                new_states.append(
                    EarleyState(
                        rule=p_prim,
                        rule_prob=candidate_state.rule_prob,
                        dot_position=candidate_state.dot_position + 1,
                        origin_position=complete_state.origin_position,
                        bbox_x=candidate_state.bbox_x,
                        bbox_y=candidate_state.bbox_y,
                        bbox_X=(
                            complete_state.bbox_X
                            if is_mu_null
                            else candidate_state.bbox_X
                        ),
                        bbox_Y=complete_state.bbox_Y,
                        scanning_history=complete_state.scanning_history,
                        child_states=(candidate_state.child_states + [complete_state]),
                    )
                )
            if p.split_direction == "horizontal" and is_rule_vertical(p_prim):
                new_states.append(
                    EarleyState(
                        rule=p_prim,
                        rule_prob=candidate_state.rule_prob,
                        dot_position=candidate_state.dot_position + 1,
                        origin_position=(complete_state.bbox_X, complete_state.bbox_y),
                        bbox_x=candidate_state.bbox_x,
                        bbox_y=candidate_state.bbox_y,
                        bbox_X=(
                            complete_state.bbox_X
                            if is_mu_null
                            else candidate_state.bbox_X
                        ),
                        bbox_Y=complete_state.bbox_Y,
                        scanning_history=complete_state.scanning_history,
                        child_states=(candidate_state.child_states + [complete_state]),
                    )
                )
            if is_rule_vertical(p) and p_prim.split_direction == "horizontal":
                new_states.append(
                    EarleyState(
                        rule=p_prim,
                        rule_prob=candidate_state.rule_prob,
                        dot_position=candidate_state.dot_position + 1,
                        origin_position=(complete_state.bbox_x, complete_state.bbox_Y),
                        bbox_x=candidate_state.bbox_x,
                        bbox_y=candidate_state.bbox_y,
                        bbox_X=complete_state.bbox_X,
                        bbox_Y=(
                            complete_state.bbox_Y
                            if is_mu_null
                            else candidate_state.bbox_Y
                        ),
                        scanning_history=complete_state.scanning_history,
                        child_states=(candidate_state.child_states + [complete_state]),
                    )
                )
            if (
                    p.split_direction == "horizontal"
                    and p_prim.split_direction == "horizontal"
            ):
                new_states.append(
                    EarleyState(
                        rule=p_prim,
                        rule_prob=candidate_state.rule_prob,
                        dot_position=candidate_state.dot_position + 1,
                        origin_position=complete_state.origin_position,
                        bbox_x=candidate_state.bbox_x,
                        bbox_y=candidate_state.bbox_y,
                        bbox_X=complete_state.bbox_X,
                        bbox_Y=(
                            complete_state.bbox_Y
                            if is_mu_null
                            else candidate_state.bbox_Y
                        ),
                        scanning_history=complete_state.scanning_history,
                        child_states=(candidate_state.child_states + [complete_state]),
                    )
                )

        return new_states

    def __parse_trees_from_states_history(
            self, states_history: dict[tuple[int, int], list[EarleyState]], lattice: Lattice
    ) -> list[ParseTree]:
        # extract complete starting states from parsing states history
        starting_states = get_starting_states_from_history(states_history)

        # compute lattice tiles cumulative dimensions sums
        lattice_heights = [imrange.mask.shape[0] for imrange in lattice.ranges[:, 0]]
        cum_heights = np.cumsum(lattice_heights)
        lattice_widths = [imrange.mask.shape[1] for imrange in lattice.ranges[0, :]]
        cum_widths = np.cumsum(lattice_widths)

        # for each complete starting state, build parse tree by following
        # the parsing path
        parse_trees: list[ParseTree] = []
        for starting_state in starting_states:
            parse_tree = ParseTree(
                start_production=starting_state.rule,
                start_production_prob=starting_state.rule_prob,
            )

            # queue of items: (state, parent_node_id)
            states_queue: list[tuple[EarleyState, UUID]] = [
                (starting_state.child_states[0], None)
            ]

            while states_queue:
                state, parent_node_id = states_queue.pop(0)

                # create ParseNode object for the state
                up = 0 if state.bbox_y == 0 else cum_heights[state.bbox_y - 1]
                down = cum_heights[state.bbox_Y - 1]
                left = 0 if state.bbox_x == 0 else cum_widths[state.bbox_x - 1]
                right = cum_widths[state.bbox_X - 1]
                box = ImgBox(
                    up=up,
                    down=down,
                    left=left,
                    right=right,
                    symbol_id=state.rule.lhs,
                    is_terminal=False,
                )
                child_shapes = [
                    get_state_pixels_shape(child_state, lattice)
                    for child_state in state.child_states
                ]
                is_terminal = False
                if state.rule.split_direction == "vertical":
                    sizes = [
                        child_shape[1] / (right - left)
                        for child_shape in child_shapes[:-1]
                    ]
                elif state.rule.split_direction == "horizontal":
                    sizes = [
                        child_shape[0] / (down - up)
                        for child_shape in child_shapes[:-1]
                    ]
                else:
                    is_terminal = True
                    closest_attr_idx = None
                if not is_terminal:
                    sizes.append(1.0 - np.sum(sizes))
                    _, closest_attr_idx = state.rule.get_closest_attribute(sizes)
                node = ParseNode(
                    symbol=self.grammar.nonterminals[state.rule.lhs],
                    box=box,
                    rule=state.rule,
                    attribute_idx=closest_attr_idx,
                    rule_prob=state.rule_prob,
                )

                # add node to the parse tree
                parse_tree.add_node(node, parent=parent_node_id)

                # add child states to the queue
                for child_state in state.child_states:
                    states_queue.append((child_state, node.identifier))

                # if state had lexical production, add terminal node
                if is_terminal:
                    terminal_box = state.rule(box, attribute_idx=0)[0]
                    terminal_node = ParseNode(
                        symbol=self.grammar.terminals[terminal_box.symbol_id],
                        box=terminal_box,
                        rule=None,
                        attribute_idx=None,
                        rule_prob=None,
                    )
                    parse_tree.add_node(terminal_node, parent=node.identifier)

            parse_trees.append(parse_tree)

        return parse_trees
