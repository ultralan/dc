from __future__ import annotations

import unittest

import torch

from uca8.postprocess import estimate_source_count_from_heatmap


class HeatmapCounterTests(unittest.TestCase):
    def test_counts_two_separated_peaks(self) -> None:
        heatmap = torch.zeros(72)
        heatmap[10] = 0.9
        heatmap[42] = 0.85
        count = estimate_source_count_from_heatmap(heatmap, max_sources=4)
        self.assertEqual(int(count.item()), 2)

    def test_suppresses_nearby_peaks(self) -> None:
        heatmap = torch.zeros(72)
        heatmap[10] = 0.9
        heatmap[12] = 0.8
        count = estimate_source_count_from_heatmap(
            heatmap,
            max_sources=4,
            min_separation_bins=4,
        )
        self.assertEqual(int(count.item()), 1)


if __name__ == "__main__":
    unittest.main()
