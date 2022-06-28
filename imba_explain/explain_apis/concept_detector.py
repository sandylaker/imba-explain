import logging
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.utils import setup_logger
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

from ..classifiers import get_module
from .pointing_game import bboxes_to_mask


class ConceptDetector:

    def __init__(self,
                 img_size: Union[int, Tuple[int, int]] = 224,
                 quantile_threshold: float = 0.99,
                 with_bboxes: bool = False,
                 count_disjoint: bool = False) -> None:
        self.classifier: Optional[nn.Module] = None
        self.hook_handle: Optional[RemovableHandle] = None

        self._feat_maps: List[torch.Tensor] = []

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        elif isinstance(img_size, Sequence) and len(img_size) == 2:
            pass
        else:
            raise ValueError(f'Invalid img_size: {img_size}.')
        self.img_size: Tuple[int, int] = img_size
        self.quantile_threshold = quantile_threshold
        self.with_bboxes = with_bboxes
        self.count_disjoint = count_disjoint
        # map class indices to a list of integers,
        # each integer represents the number of concepts in a single sample
        self.num_concepts_dict = defaultdict(list)

    def set_classifier(
        self,
        classifier: nn.Module,
        target_layer: str,
    ) -> None:
        self.classifier = classifier
        self.classifier.eval()
        layer = get_module(classifier, target_layer)

        def _forward_hook(module, _input, _output):
            self._feat_maps.append(_output.detach())
            return _output

        self.hook_handle = layer.register_forward_hook(_forward_hook)

    def clear_feat_maps(self) -> None:
        self._feat_maps.clear()

    def reset_num_concepts(self):
        self.num_concepts_dict.clear()

    def remove_classifier_and_hook(self) -> None:
        self.classifier = None
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def report(self, ind_to_name: Dict, logger: Optional[logging.Logger] = None) -> None:

        logger = setup_logger('imba-explain') if logger is None else logger
        round_prec = 4

        global_num_concepts = list(chain.from_iterable(self.num_concepts_dict.values()))
        global_min_val = np.min(global_num_concepts)
        global_max_val = np.max(global_num_concepts)
        global_mean_val = np.round(np.mean(global_num_concepts), round_prec)
        global_std_val = np.round(np.std(global_num_concepts), round_prec)

        class_names = [ind_to_name[k] for k in self.num_concepts_dict.keys()]
        # append the row name '[Global]' to the class names
        class_names.append('[Global]')

        cls_wise_min_vals = [np.min(v) for v in self.num_concepts_dict.values()]
        cls_wise_max_vals = [np.max(v) for v in self.num_concepts_dict.values()]
        cls_wise_mean_vals = [np.round(np.mean(v), round_prec) for v in self.num_concepts_dict.values()]
        cls_wise_std_vals = [np.round(np.std(v), round_prec) for v in self.num_concepts_dict.values()]

        # append global statistics to the class-wise statistics
        cls_wise_min_vals.append(global_min_val)
        cls_wise_max_vals.append(global_max_val)
        cls_wise_mean_vals.append(global_mean_val)
        cls_wise_std_vals.append(global_std_val)

        tabular_data = {
            'Classes': class_names,
            'Min': cls_wise_min_vals,
            'Max': cls_wise_max_vals,
            'Mean': cls_wise_mean_vals,
            'Std': cls_wise_std_vals
        }

        table = tabulate(tabular_data, headers='keys', tablefmt='pretty', numalign='left', stralign='left')
        log_str = 'Number of Concepts:\n'
        log_str += f'{table}'
        logger.info(log_str)

    def detect(self,
               data_loader: DataLoader,
               device: Union[str, torch.device] = 'cuda:0',
               with_pbar: bool = False,
               remove_classifier_and_hook=True) -> None:
        pbar = tqdm(total=len(data_loader)) if with_pbar else None

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                img = batch['img'].to(device)
                # labels: [[0, 3, ...], [1, 2, ...]]
                # outer list denotes a batch of samples,
                # inner list denotes class indices of objects present in each sample
                labels = batch['labels']
                if self.with_bboxes:
                    if 'bboxes' not in batch:
                        raise KeyError("When 'self.with_bboxes' is True, 'bboxes' should be in the data batch.")
                    # bboxes in a batch data is a list, where each element is a 2-D ndarray
                    bboxes: List[np.ndarray] = batch['bboxes']
                else:
                    bboxes = None

                _ = self.classifier(img)

                feat_map = self._feat_maps[0]
                num_samples, num_channels, feat_h, feat_w = feat_map.shape

                feat_map = F.interpolate(feat_map, size=self.img_size, mode='bilinear')
                # feat_map: (num_channels, num_samples, img_h, img_w)
                feat_map = torch.permute(feat_map, [1, 0, 2, 3])
                # feat_map: (num_channels, num_samples * img_h * img_w)
                feat_map = feat_map.reshape(num_channels, num_samples * self.img_size[0] * self.img_size[1]).cpu()
                threshold = torch.quantile(feat_map, q=self.quantile_threshold, dim=1, keepdim=True)
                assert threshold.shape == (num_channels, 1)

                binary_mask = (feat_map >= threshold).to(torch.float32)
                binary_mask = binary_mask.reshape(num_channels, num_samples, self.img_size[0], self.img_size[1])
                binary_mask = torch.permute(binary_mask, [1, 0, 2, 3])
                assert binary_mask.shape == (num_samples, num_channels) + self.img_size

                # binary_mask: (num_samples, num_channels, img_h, img_w)
                binary_mask = binary_mask.numpy()
                binary_mask = (binary_mask * 255.0).astype(np.uint8)
                if bboxes is not None:
                    # convert bboxes for each sample into a 2-D binary mask, and concatenate them,
                    # the concatenated bboxes_binary_mask has shape: (num_samples, 1, img_h, img_w)
                    bboxes_binary_mask = [bboxes_to_mask(b, shape=self.img_size, dtype=np.uint8) for b in bboxes]
                    bboxes_binary_mask = np.stack(bboxes_binary_mask)[:, None, :, :]
                    binary_mask *= bboxes_binary_mask

                # mask_single_sample: (num_channels, img_h, img_w)
                for mask_single_sample, labels_single_sample in zip(binary_mask, labels):
                    num_concepts_single_sample = 0
                    for mask_single_channel in mask_single_sample:
                        contours, _ = cv2.findContours(mask_single_channel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if self.count_disjoint:
                            num_concepts_single_sample += len(contours)
                        else:
                            num_concepts_single_sample += (len(contours) > 0)

                    # the num_concepts_single_samples is shared by multiple labels of one single sample
                    for label in labels_single_sample:
                        self.num_concepts_dict[label].append(num_concepts_single_sample)
                    if pbar is not None:
                        pbar.set_postfix({'num_concepts': num_concepts_single_sample})
                        pbar.refresh()

                self.clear_feat_maps()

                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        if remove_classifier_and_hook:
            self.remove_classifier_and_hook()
