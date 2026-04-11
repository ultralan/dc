from __future__ import annotations

import unittest

import torch

from uca8.geometry.uca8 import azimuth_grid, default_mic_pairs, make_uniform_circular_array


class GeometryTests(unittest.TestCase):
    def test_uniform_circular_array_has_constant_radius(self) -> None:
        geometry = make_uniform_circular_array(num_mics=8, radius=0.045)
        self.assertEqual(tuple(geometry.shape), (8, 3))
        radius = torch.linalg.norm(geometry[:, :2], dim=-1)
        self.assertTrue(torch.allclose(radius, torch.full_like(radius, 0.045), atol=1e-5))

    def test_default_pairs_match_expected_cardinality(self) -> None:
        pairs = default_mic_pairs(8)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(pairs[0], (0, 1))
        self.assertEqual(pairs[-1], (3, 7))

    def test_azimuth_grid_size(self) -> None:
        grid = azimuth_grid(72)
        self.assertEqual(grid.numel(), 72)
        self.assertLess(float(grid.max()), 3.2)


if __name__ == "__main__":
    unittest.main()
