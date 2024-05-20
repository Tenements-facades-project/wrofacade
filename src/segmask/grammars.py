import json
from typing import Any
import pickle
from PIL import Image
from pydantic import ValidationError
from .seg_mask import SegMaskGenerator
from src.utils.grammars_utils.ascfg import Grammar
from src.utils.segmentation_mask import SegmentationMask


class GrammarMaskGenerator(SegMaskGenerator):
    """Class for generating new segmentation mask using
    trained stochastic grammar (ASCFG)

    Attributes:
        grammar: Grammar object to generate masks with
        label2clr: dictionary containing class labels and colours, keys' positional index
            in the dict corresponds to the class id, e.g. {'window': [255,244,233]}
        n_attempts: maximum number of attempts to generate mask

    """

    def __init__(
        self,
        hp: dict[str, Any],
    ):
        super().__init__()
        self.__load_grammar_from_file(hp["grammars_masks"]["grammar_pickle_path"])
        with open(hp["label2clr_path"]) as f:
            self.label2clr: dict[str, list[int]] = json.load(f)
        self.n_attempts: int = hp["grammars_masks"]["n_attempts"]

    def __load_grammar_from_file(self, file_path: str):
        """Loads Grammar object from pickle
        and saves it in `self.grammar`
        """
        with open(file_path, "rb") as f:
            grammar = pickle.load(f)
        if not isinstance(grammar, Grammar):
            raise RuntimeError("Provided pickle file does not contain grammar object!")
        self.grammar = grammar

    def generate_mask(self, args: dict) -> SegmentationMask:
        """Generates new segmentation mask using grammar

        Returns:
            SegmentationMask: generated mask object

        Raises:
            RuntimeError: when number of unsuccessful attempts
                reaches `self.n_attempts`
        """
        for _ in range(self.n_attempts):
            try:
                parse_tree = self.grammar.generate_parse_tree()
                _, mask = parse_tree.assemble_image()
                return SegmentationMask(
                    Image.fromarray(mask, "L"),
                    label2clr=self.label2clr,
                )
            except ValidationError:
                pass
            except ValueError:
                pass
        raise RuntimeError(
            "Maximum number of unsuccessful attempts reached when trying"
            "to generate mask"
        )
