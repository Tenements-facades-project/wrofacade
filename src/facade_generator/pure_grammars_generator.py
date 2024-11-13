import pickle

from src.facade_generator.facade_generator_abc import FacadeGenerator
from src.config.hparam import hparam
from src.utils.grammars_utils.ascfg import Grammar


class PureGrammarGenerator(FacadeGenerator):
    """Generates images just with a grammar

    Args:
        grammar_pickle_path: path to the pickle with the grammar object
            (by default, value is read from the config yaml)

    """

    def __init__(
        self, grammar_pickle_path: str = hparam["grammars_pure"]["grammar_pickle_path"]
    ):
        self.__load_grammar_from_file(grammar_pickle_path)

    def __load_grammar_from_file(self, file_path: str):
        """Loads Grammar object from pickle
        and saves it in `self.grammar`
        """
        with open(file_path, "rb") as f:
            grammar = pickle.load(f)
        if not isinstance(grammar, Grammar):
            raise RuntimeError("Provided pickle file does not contain grammar object!")
        self.grammar = grammar

    def generate_facade(self, params: dict = None):
        """Generates new image using grammar

        Args:
            params (dict): keys:
                n_floors (int, Optional): number of floors in the generated facade
                    (including ground floor)
                min_height_width_ratio (float, Optional): minimum height / width ratio
                    of the generated image
                max_height_width_ratio (float, Optional): maximum height / width ratio
                    of the generated image

        Returns:
            array: generated image

        Notes:
            due to inaccurate parsing tree creation, grammar sometimes fails
                to create facade with desired number of floors
        """
        params = {} if params is None else params

        n_floors = params.get("n_floors", None)
        min_height_width_ratio = params.get("min_height_width_ratio", None)
        max_height_width_ratio = params.get("max_height_width_ratio", None)

        rules_choosers = dict()
        if n_floors:
            rules_choosers[1] = lambda rule: len(rule.rhs) == (n_floors + 1)
        if min_height_width_ratio or max_height_width_ratio:
            rule_min_ratio = (
                (
                    lambda rule: rule.start_shape[0] / rule.start_shape[1]
                    >= min_height_width_ratio
                )
                if min_height_width_ratio
                else lambda rule: True
            )
            rule_max_ratio = (
                (
                    lambda rule: rule.start_shape[0] / rule.start_shape[1]
                    <= max_height_width_ratio
                )
                if max_height_width_ratio
                else lambda rule: True
            )
            rules_choosers[0] = lambda rule: rule_min_ratio(rule) and rule_max_ratio(
                rule
            )

        parse_tree = self.grammar.generate_parse_tree(rules_choosers=rules_choosers)
        img, _ = parse_tree.assemble_image()
        return img[:, :, ::-1]
