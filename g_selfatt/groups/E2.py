import math

import numpy as np
import torch

from .group import Group

GroupElement = torch.Tensor


class E2(Group):
    def __init__(self, num_elements):
        """
        Elements of the E2 group are specified by [rotation, mirroring ]. The identity of this group is [0, 1].
        For instance, the element [pi / 2, -1] corresponds to a rotation by 90 degrees followed by a permutation along the y axis.
        """
        super().__init__()

        self.register_buffer("_identity", torch.from_numpy(np.array([0.0, 1.0], dtype=np.float32)))
        self.register_buffer("_elements", self.discretize_group(num_elements))
        self.register_buffer("_relative_positions", self.construct_relative_positions())

    def left_action_on_Rd(self, h: GroupElement, pp: torch.Tensor):
        if False not in (h == self.identity):
            return pp
        else:
            # Fist rotate, then reflect over the y axis
            Lhpp = torch.einsum("oi,ixy->oxy", self.matrix_form(h[0]), pp).type(torch.float32)
            if h[-1] == -1:
                Lhpp = torch.flip(Lhpp, dims=[-1])
            return Lhpp

    def left_action_on_H(self, h: GroupElement, pp_h: torch.Tensor):
        if False not in (h == self.identity):
            return pp_h
        else:
            Lhpp_h = pp_h.clone().view(2, self.num_elements // 2, -1)
            # They rotate in opposite directions
            if h[0] != 0:
                shift = int(
                    torch.round((1.0 / ((2 * np.pi) / (self.num_elements // 2)) * -h[0])).item()
                )
                Lhpp_h[0, :, :] = torch.roll(Lhpp_h[0, :, :], shifts=shift, dims=-2)
                Lhpp_h[1, :, :] = torch.roll(Lhpp_h[1, :, :], shifts=-shift, dims=-2)
            if h[-1] == -1:
                Lhpp_h = torch.roll(Lhpp_h, shifts=1, dims=0)
            return Lhpp_h.view(pp_h.shape)

    def absolute_determinant(self, h: GroupElement):
        return 1.0

    def discretize_group(self, num_elements: int):
        # Recall that E2 group consists of 2 elements: the element of rotations r and the element of reflections m.
        num_elements_r = int(num_elements / 2)

        h_list_r = np.array(
            [
                np.linspace(
                    0, 2 * math.pi * float(num_elements_r - 1) / num_elements_r, num_elements_r
                )
            ],
            dtype=np.float64,  # Required for precise rotation.
        ).transpose()
        h_list_m = np.stack(
            (
                (np.concatenate((h_list_r, h_list_r), axis=0)).squeeze(),  # 2 times rotation
                np.concatenate(
                    (
                        np.ones(num_elements_r, dtype=np.float64),
                        -1 * np.ones(num_elements_r, dtype=np.float64),
                    )
                ).transpose(),
            ),  # [1, ..., -1, ...]
            axis=1,
        )
        return torch.from_numpy(h_list_m)

    def construct_relative_positions(self):
        indices = torch.arange(self.num_elements)
        return torch.stack(
            [self.left_action_on_H(torch.tensor([-h[0], h[1]]), indices) for h in self.elements],
            dim=0,
        )

    def matrix_form(self, h: torch.Tensor) -> torch.Tensor:
        """
        Converts the element of the group h into a matrix with which the action of the group can be applied to a vector
        by means of multiplication.

        For example, for a rotation of 90 degrees, it transforms the element h = pi into a matrix:

            [math.cos(pi), -math.sin(pi)],
            [math.sin(pi), math.cos(pi)]

        which transforms an input [x , y] into a rotated vector [x_new, y_new] by means of rotation.
        """
        angle = h.item()
        mat_repr = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        return mat_repr
