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
        parse_tree = self.grammar.generate_parse_tree()
        img, _ = parse_tree.assemble_image()
        return img[:, :, ::-1]
