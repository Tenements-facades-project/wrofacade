import os
import pickle
import uuid
from uuid import UUID
import numpy as np
import pandas as pd
from tqdm import tqdm

from .lattice import Lattice
from .ascfg import Nonterminal, Grammar
from .earley_parsing import EarleyParser2D


class BayesianMerger:
    """Class implementing Bayesian merging for grammars

    Attributes:
        w: w parameter of the loss function
        strategy ({"random_draw"}): strategy used for grammars space search
        n_random_draws: number of random draws in each epoch;
            used in strategy "random_draw"
        eps: smoothing term used while computing logarithms
    """

    def __init__(self, w: float, strategy: str, n_random_draws: int = 10, eps: float = 1e-4):
        self.w: float = w
        self.strategy: str = strategy
        self.n_random_draws: int = n_random_draws
        self.eps: float = eps

    @staticmethod
    def __get_nonterminals_df(grammar: Grammar) -> pd.DataFrame:
        """For given grammar, builds dataframe where each row
        corresponds to one grammar's nonterminal, where nonterminals
        UUIDs are indices
        """

        nonterminals_ids: list[UUID] = []
        nonterminals_list: list[Nonterminal] = []
        nonterminals_reachable: list[set[int]] = []

        for nonterm_id, nonterm in grammar.nonterminals.items():
            nonterminals_ids.append(nonterm_id)
            nonterminals_list.append(nonterm)
            nonterminals_reachable.append(nonterm.reachable_labels)

        nonterminals_df = pd.DataFrame(
            {'nonterminals': nonterminals_list,
            'reachable_labels': nonterminals_reachable},
            index=nonterminals_ids,
        )
        return nonterminals_df

    @staticmethod
    def __get_random_nonterminal(nonterminals_df: pd.DataFrame) -> tuple[UUID, Nonterminal]:
        """Given a dataframe of nonterminals, draws random
        nonterminal
        """
        sample = nonterminals_df.sample()
        nonterm_id = sample.index[0]
        nonterm = sample["nonterminals"].iloc[0]
        return nonterm_id, nonterm

    def __random_merge(self, grammar: Grammar) -> Grammar:
        """Draws two random label-compatible nonterminals
        from the grammar and returns new grammar with these
        two nonterminals merged into one nonterminal
        """
        nonterminals_df = self.__get_nonterminals_df(grammar)
        nonterm_1_id, nonterm_1 = self.__get_random_nonterminal(nonterminals_df)

        same_reachable_df = nonterminals_df[
            nonterminals_df["reachable_labels"] == nonterm_1.reachable_labels]
        same_reachable_df = same_reachable_df.drop(nonterm_1_id)
        if same_reachable_df.shape[0] == 0:
            # no compatible nonterminal
            # try once again
            return self.__random_merge(grammar)
        nonterm_2_id, nonterm_2 = self.__get_random_nonterminal(same_reachable_df)

        new_nonterm_id = uuid.uuid4()

        grammar = grammar.merge_nonterminals(
            nonterm_1_id=nonterm_1_id, nonterm_2_id=nonterm_2_id, new_nonterm_id=new_nonterm_id
        )

        return grammar

    @staticmethod
    def __grammar_dl(grammar: Grammar) -> float:
        """Calculates Description Length for
        given grammar
        """
        return np.log(len(grammar.nonterminals))

    def __lattice_logprob(self, lattice: Lattice, parser: EarleyParser2D) -> float:
        """Calculates loglikelihood of the given lattice,
        i.e. logarithm of sum of all parse trees probabilities
        that are generated with given grammar and generate
        the lattice
        """

        # get all grammar parse trees that generate the lattice
        parse_trees, _ = parser.parse_lattice(lattice)

        # sum all parse trees probabilities and return log of it
        return np.log(
            np.sum([
                np.exp(parse_tree.logprob()) for parse_tree in parse_trees
            ]
                  ) + self.eps
        )

    def __grammar_loss(
        self,
        grammar: Grammar,
        parser: EarleyParser2D,
        lattices: list[Lattice],
        w: float,
    ) -> float:
        logprob = sum(
            self.__lattice_logprob(lattice=lattice, parser=parser)
            for lattice in lattices
        )
        dl = self.__grammar_dl(grammar)
        return w * dl - logprob

    def __random_draw_epoch(
        self,
        grammar: Grammar,
        lattices: list[Lattice],
    ) -> tuple[Grammar, float]:

        new_grammars = [
            self.__random_merge(grammar) for _ in range(self.n_random_draws)
        ]

        new_grammars_losses = [
            self.__grammar_loss(
                new_grammar, parser=EarleyParser2D(new_grammar), lattices=lattices, w=self.w
            ) for new_grammar in tqdm(new_grammars)
        ]

        best_idx = np.argmin(new_grammars_losses)

        return new_grammars[best_idx], new_grammars_losses[best_idx]

    def perform_merging(
        self,
        grammar: Grammar,
        lattices: list[Lattice],
        n_epochs: int,
        checkpoint_dir: str | None = None,
        checkpoint_every_n: int | None = None
    ) -> tuple[Grammar, float]:
        for i in range(n_epochs):
            print(f"Epoch {i + 1}")
            try:
                if self.strategy == "random_draw":
                    grammar, loss_val = self.__random_draw_epoch(
                        grammar=grammar, lattices=lattices
                    )
                else:
                    raise AssertionError("Invalid `strategy` value!")
                print(f"Loss: {loss_val}")
            except Exception as e:
                print("ERROR")
                print(e)
            if checkpoint_dir:
                if i % checkpoint_every_n == 0:
                    with open(os.path.join(checkpoint_dir, f"checkpoint_{i}.pickle"), 'wb') as f:
                        pickle.dump(grammar, f)
        return grammar, loss_val
