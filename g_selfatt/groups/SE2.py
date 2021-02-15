import math

import numpy as np
import torch

from .group import Group

GroupElement = torch.Tensor


class SE2(Group):
    def __init__(self, num_elements):
        super().__init__()

        self.register_buffer("_identity", torch.from_numpy(np.array([0.0], dtype=np.float32)))
        self.register_buffer("_elements", self.discretize_group(num_elements))
        self.register_buffer("_relative_positions", self.construct_relative_positions())

    def left_action_on_Rd(self, h: GroupElement, pp: torch.Tensor):
        if h == self.identity:
            return pp
        else:
            return torch.einsum("oi,ixy->oxy", self.matrix_form(h), pp).type(torch.float32)

    def left_action_on_H(self, h: GroupElement, pp_h: torch.Tensor):
        if h == self.identity:
            return pp_h
        else:
            shift = int(torch.round((1.0 / ((2 * np.pi) / self.num_elements) * -h)).item())
            return torch.roll(pp_h, shifts=shift, dims=-1)

    def absolute_determinant(self, h: GroupElement):
        return 1.0

    def discretize_group(self, num_elements: int):
        return torch.from_numpy(
            np.array(
                [
                    np.linspace(
                        0, 2 * math.pi * float(num_elements - 1) / float(num_elements), num_elements
                    )
                ],
                dtype=np.float64,  # Required for precise rotation.
            ).transpose()
        )

    def construct_relative_positions(self):
        indices = torch.arange(self.num_elements)
        return torch.stack([self.left_action_on_H(-h, indices) for h in self.elements], dim=0)

    def matrix_form(self, h: torch.Tensor) -> torch.Tensor:
        """
        Converts the element of the group h into a matrix with which the action of the group can be applied to a vector
        by means of multiplication.

        For example, for a rotation of 90 degrees, it transforms the element h = pi / 2 into a matrix:

            [math.cos(pi / 2), -math.sin(pi / 2)],
            [math.sin(pi / 2), math.cos(pi / 2)]

        which transforms an input [x , y] into a rotated vector [x_new, y_new] by means of rotation.
        """
        angle = h.item()
        mat_repr = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        return mat_repr
