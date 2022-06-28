import os.path as osp
from typing import Dict, List, Optional, Union

import cv2
import mmcv
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .builder import DATASETS
from .cls_dataset import ClassificationDataset


@DATASETS.register_module()
class CheXpertDataset(ClassificationDataset):
    name_to_ind = {
        'Atelectasis': 0,
        'Cardiomegaly': 1,
        'Consolidation': 2,
        'Edema': 3,
        'Enlarged Cardiomediastinum': 4,
        'Fracture': 5,
        'Lung Lesion': 6,
        'Lung Opacity': 7,
        'No Finding': 8,
        'Pleural Effusion': 9,
        'Pleural Other': 10,
        'Pneumonia': 11,
        'Pneumothorax': 12,
        'Support Devices': 13,
    }

    def __init__(self,
                 data_root: str,
                 label_csv: str,
                 pipeline: List[Dict],
                 indices_file: Optional[str] = None,
                 clip_ratio: Optional[float] = None,
                 select_classes: Optional[List[Union[str, int]]] = None) -> None:
        super(CheXpertDataset, self).__init__(pipeline=pipeline, clip_ratio=clip_ratio, select_classes=select_classes)

        self.data_root = data_root

        self.label_csv = pd.read_csv(label_csv)
        self.label_csv.fillna(0, inplace=True)
        self.label_csv.replace({-1: 0}, inplace=True)
        self.label_csv['relpath'] = self.label_csv['Path'].apply(lambda x: x.split('/', maxsplit=1)[1])
        self.label_csv.drop(['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA'], axis=1, inplace=True)

        indices = np.array(mmcv.list_from_file(indices_file), dtype=int) if indices_file is not None \
            else np.arange(len(self.label_csv))
        is_selected = (np.count_nonzero(self.label_csv.loc[indices, self._select_class_names], axis=1) != 0)
        self.indices = indices[is_selected]

        self.num_pos_neg = torch.zeros((2, self._num_classes), dtype=torch.long)
        self.num_pos_neg[0] = torch.tensor(
            self.label_csv.loc[self.indices, self._class_names].values.sum(0), dtype=torch.long)
        self.num_pos_neg[1] = len(self.indices) - self.num_pos_neg[0]
        self.log_data_dist_info(class_names=self._class_names)

    def get_num_pos_neg(self) -> torch.Tensor:
        return self.num_pos_neg

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        img_file = self.label_csv.loc[idx, 'relpath']
        one_hot_label = self.label_csv.loc[idx, self._class_names].values.astype(int)
        one_hot_label = torch.tensor(one_hot_label, dtype=torch.float32)

        img = cv2.cvtColor(cv2.imread(osp.join(self.data_root, img_file)), cv2.COLOR_BGR2RGB)
        img = self.pipeline(image=img)['image']
        result = {'img': img, 'img_file': img_file, 'target': one_hot_label}
        return result
