"""Preprocessing exports.

Keep the plotting-oriented ``TPPDataLoader`` import lazy.  The model-facing
pickle path only needs the tokenizer and dataset, and should not require
Matplotlib merely because this package is imported.
"""

from easy_tpp.preprocess.dataset import TPPDataset, get_data_loader
from easy_tpp.preprocess.event_tokenizer import EventTokenizer

__all__ = [
    "TPPDataLoader",
    "EventTokenizer",
    "TPPDataset",
    "get_data_loader",
]


def __getattr__(name):
    if name == "TPPDataLoader":
        from easy_tpp.preprocess.data_loader import TPPDataLoader

        return TPPDataLoader
    raise AttributeError(name)
