from typing import Optional, Sequence

import numpy as np

from .pointing_game import bboxes_to_mask


class Overlap:

    def __init__(self, mode: str = 'iobb', attr_threshold: Optional[float] = None) -> None:
        if mode.lower() not in ('iobb', 'ior', 'iou'):
            raise ValueError(f"mode should be in ('iobb', 'ior'), but got {mode}")
        if attr_threshold is not None and (attr_threshold >= 1.0 or attr_threshold <= 0.0):
            raise ValueError(f"'attr_threshold' should be in range (0, 1), but got {attr_threshold}")

        self.mode = mode
        self.attr_threshold = attr_threshold

    def evaluate(self, attr_map: np.ndarray, bboxes: Sequence[np.ndarray]) -> float:
        if len(attr_map.shape) != 2:
            raise ValueError(f'attribution map should be in shape (height, width), but got {attr_map.shape}')
        if np.max(attr_map) > 1.0 or np.min(attr_map) < 0.0:
            raise ValueError(f'attribution map should have value range [0, 1], '
                             f'but got [{np.min(attr_map):.4f}, {np.max(attr_map):.4f}]')

        gt_mask = bboxes_to_mask(bboxes, shape=attr_map.shape, dtype=float)
        if self.attr_threshold is not None:
            attr_map = (attr_map >= self.attr_threshold).astype(gt_mask.dtype)

        inter = (gt_mask * attr_map).sum()
        if self.mode == 'iobb':
            iobb = inter / (gt_mask.sum() + 1e-8)
            return iobb
        elif self.mode == 'ior':
            ior = inter / (attr_map.sum() + 1e-8)
            return ior
        else:
            iou = inter / (gt_mask.sum() + attr_map.sum() - inter + 1e-8)
            return iou
