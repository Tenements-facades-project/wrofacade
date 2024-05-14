from .lattice import Lattice


def crop_facade(
    lattice: Lattice, background_label: int, cutoff_threshold: float = 0.7
) -> Lattice:
    """Returns new lattice object with removed
    ranges rows and columns that contain mostly
    background ranges

    Args:
        lattice (Lattice): lattice of a facade with background ranges
        background_label (int): label of the background pixel class
        cutoff_threshold (float): value from range (0-1),
            maximum part of background ranges

    Returns:
        lattice with background cut off
    """
    rows_indicator = []
    cols_indicator = []
    n_rows, n_cols = lattice.ranges.shape
    for row in lattice.ranges:
        rows_indicator.append(
            sum([rng.most_frequent_label() == background_label for rng in row]) < n_cols * cutoff_threshold
        )
    for col in lattice.ranges.transpose():
        cols_indicator.append(
            sum([rng.most_frequent_label() == background_label for rng in col]) < n_rows * cutoff_threshold
        )
    return lattice[rows_indicator][:, cols_indicator]
