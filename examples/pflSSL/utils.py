"""
The necessary tools for self-supervised learning approaches.

"""
from typing import List, Tuple

from torch import Tensor
from lightly.data.multi_view_collate import MultiViewCollate


class MultiViewCollateWrapper(MultiViewCollate):
    """ An interface to connect the collate from lightly with the data loading schema of 
     Plato. """
    def __call__(
        self, batch: List[Tuple[List[Tensor], int, str]]
    ) -> Tuple[List[Tensor], Tensor, List[str]]:
        views, labels, _ = super().__call__(batch)

        return views, labels
