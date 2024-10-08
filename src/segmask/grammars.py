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

        Args:
            args (dict): keys:
                n_floors (int, Optional): number of floors in the generated facade
                    (including ground floor)
                min_height_width_ratio (float, Optional): minimum height / width ratio
                    of the generated image
                max_height_width_ratio (float, Optional): maximum height / width ratio
                    of the generated image

        Returns:
            SegmentationMask: generated mask object

        Notes:
            due to inaccurate parsing tree creation, grammar sometimes fails
                to create facade with desired number of floors
        """
        n_floors = args.get("n_floors", None)
        min_height_width_ratio = args.get("min_height_width_ratio", None)
        max_height_width_ratio = args.get("max_height_width_ratio", None)

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
        _, mask = parse_tree.assemble_image()
        return SegmentationMask(
            Image.fromarray(mask, "L"),
            label2clr=self.label2clr,
        )
