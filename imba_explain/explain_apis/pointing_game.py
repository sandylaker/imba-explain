from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np


def bboxes_to_mask(bboxes: Sequence[np.ndarray], shape: Tuple[int, int], dtype: Union[Any] = np.bool) -> np.ndarray:
    binary_mask = np.zeros(shape, dtype=dtype)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        binary_mask[y1:y2, x1:x2] = 1

    return binary_mask


class PointingGame:

    def __init__(self, top_k: int = 1) -> None:
        self.top_k = top_k

    def evaluate(self, attr_map: np.ndarray, bboxes: Sequence[np.ndarray]) -> Dict[str, float]:
        if len(attr_map.shape) != 2:
            raise ValueError(f'attribution map should be in shape (height, width), but got {attr_map.shape}')

        gt_mask = bboxes_to_mask(bboxes, shape=attr_map.shape).flatten()
        attr_map = attr_map.flatten()
        kth_ind = np.argpartition(attr_map, -self.top_k)[-self.top_k]

        threshold = attr_map[kth_ind]
        if threshold == 0:
            # the case where the attr_map is zeros
            return {'num_true_pos': 0, 'num_pos': 0}

        is_top_k = attr_map >= threshold
        num_true_pos = (is_top_k * gt_mask).sum()
        num_pos = is_top_k.sum()
        return {'num_true_pos': num_true_pos, 'num_pos': num_pos}
