from typing import Generic, Tuple, TypeVar

import torch.nn as nn
from torch import Tensor

GroupElement = TypeVar("GroupElement")


class Group(nn.Module, Generic[GroupElement]):
    def __init__(self):
        """
        Boilerplate for the construction of a Group class.

        Subclasses of this class must implement the functions:
        left_action_on_Rd(self, h: GroupElement, pp: Tensor) -> Tensor
        left_action_on_H(self, h: GroupElement, pp: Tensor) -> Tensor
        discretize_group(self, num_elements: int) -> Tensor
        absolute_determinant(self, h: GroupElement) -> float

        And define the class variables:
        self._identity
        self._elements
        self._relative_positions

        For more information on the functionality of these functions, see the documentation in the corresponding function.
        """
        super().__init__()

    def left_action_on_Rd(self, h: GroupElement, pp: Tensor) -> Tensor:
        """
        Defines how a group element h acts on a tensor of positions pp,  and returns the acted vector pp_acted = action of h on pp.
        """
        raise NotImplementedError()

    def left_action_on_H(self, h: GroupElement, pp: Tensor) -> Tensor:
        """
        Defines how a group element h acts on another group element pp,  and returns the acted element pp_acted = action of h on pp.
        """
        raise NotImplementedError()

    def discretize_group(self, num_elements: int) -> Tensor:
        """
        Discretizes the underlying continuous group to form a discrete group with num_elements elements. For instance, for the
        SE(2) group, the group of planar rotations and num_elements = 4, it creates a group of rotations by 90 degrees.
        """
        raise NotImplementedError()

    def absolute_determinant(self, h: GroupElement) -> float:
        """
        Computes the determinant of the Jacobian of the action of the group element h. This is 1 for E(2) and SE(2), but for groups such as
        scaling, this can be different for different group elements.
        """
        raise NotImplementedError()

    def left_action_on_G(self, h: GroupElement, pp: Tensor, pp_h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the action of a group element h on a member of the group G. The function returns a tuple with the action of h on the
        spatial positions in the first position, and the action of h on the group elements h in the second:

        Returns: [ left_action_on_Rd(h, pp), left_action_on_H(h, pp_h) ]
        """
        return self.left_action_on_Rd(h, pp), self.left_action_on_H(h, pp_h)

    def construct_relative_positions(self) -> Tensor:
        """
        Constructs a vector of relative positions on the group elements h.
        """
        raise NotImplementedError()

    @property
    def elements(self) -> Tensor:
        return self._elements

    @property
    def num_elements(self):
        return len(self._elements)

    @property
    def relative_positions(self):
        return self._relative_positions

    @property
    def identity(self):
        return self._identity

    def check_element(self, element: GroupElement):
        if element not in self._elements:
            raise ValueError(f"Unknown group element '{element}'.")
