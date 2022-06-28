import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .builder import DATASETS, build_pipeline
from .cls_dataset import ClassificationDataset
from .utils import load_bbox_annot


@DATASETS.register_module()
class NIHClassificationDataset(ClassificationDataset):
    name_to_ind = {
        'Atelectasis': 0,
        'Cardiomegaly': 1,
        'Consolidation': 2,
        'Edema': 3,
        'Effusion': 4,
        'Emphysema': 5,
        'Fibrosis': 6,
        'Hernia': 7,
        'Infiltration': 8,
        'Mass': 9,
        'No Finding': 10,
        'Nodule': 11,
        'Pleural_Thickening': 12,
        'Pneumonia': 13,
        'Pneumothorax': 14
    }

    def __init__(self,
                 img_root: str,
                 label_csv: str,
                 pipeline: List[Dict],
                 clip_ratio: Optional[float] = None,
                 select_classes: Optional[List[Union[str, int]]] = None) -> None:
        super().__init__(pipeline=pipeline, clip_ratio=clip_ratio, select_classes=select_classes)

        self.img_root = img_root
        self.label_csv = label_csv

        label_df = pd.read_csv(self.label_csv)
        label_df.set_index('Image Index', inplace=True)

        self.num_pos_neg = torch.zeros((2, self._num_classes), dtype=torch.long)
        # img_file to disease names. E.g., 0001_0000.png -> ['Edema', 'Pneumonia']
        self.img_to_diseases = {}
        all_img_files = os.listdir(self.img_root)
        # img files after being filtered
        self.img_files = []
        for img_file in all_img_files:
            diseases = label_df.loc[img_file, 'Finding Labels'].split('|')
            diseases = [d for d in diseases if d in self._select_class_names]
            # drop the sample if no label of selective classes is found
            if len(diseases) > 0:
                self.img_to_diseases.update({img_file: diseases})
                self.num_pos_neg[0] += self.one_hot_encode(self.name_to_ind, diseases, dtype=torch.long)
                self.img_files.append(img_file)

        self.num_pos_neg[1] = len(self.img_files) - self.num_pos_neg[0]
        self.log_data_dist_info(class_names=self._class_names)

    def get_num_pos_neg(self) -> torch.Tensor:
        return self.num_pos_neg

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        img_file = self.img_files[idx]
        img = cv2.cvtColor(cv2.imread(osp.join(self.img_root, img_file)), cv2.COLOR_BGR2RGB)
        img = self.pipeline(image=img)['image']
        result = {'img': img, 'img_file': img_file}

        diseases = self.img_to_diseases[img_file]
        one_hot_label = self.one_hot_encode(self.name_to_ind, diseases)
        result.update({'target': one_hot_label})

        return result


@DATASETS.register_module()
class NIHBinaryClassificationDataset(ClassificationDataset):
    # the index only indicates the colum index in the output/target (shape: (batch_size, num_classes)),
    # but not the positive target value
    name_to_ind = {'Unhealthy': 0}

    def __init__(self, img_root: str, label_csv: str, pipeline: List[Dict], clip_ratio: Optional[float] = None) -> None:
        super().__init__(pipeline=pipeline, clip_ratio=clip_ratio)

        self.img_root = img_root
        self.label_csv = label_csv

        label_df = pd.read_csv(self.label_csv)
        label_df.set_index('Image Index', inplace=True)

        self.num_pos_neg = torch.zeros((2, self._num_classes), dtype=torch.long)
        # img_file to disease names. E.g., 0001_0000.png -> 0 (No finding) or 1 (others)
        self.img_to_label = {}
        all_img_files = os.listdir(self.img_root)

        self.samples: List[Tuple[str, int]] = []
        for img_file in all_img_files:
            diseases = label_df.loc[img_file, 'Finding Labels'].split('|')
            if 'No Finding' not in diseases:
                self.samples.append((img_file, 1))
                # 'Pathology' is positive class and 'No Finding' is negative class.
                self.num_pos_neg[0] += 1
            else:
                self.samples.append((img_file, 0))

        self.num_pos_neg[1] = len(self.samples) - self.num_pos_neg[0]
        self.log_data_dist_info(class_names=self._class_names)

    def get_num_pos_neg(self) -> torch.Tensor:
        return self.num_pos_neg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        img_file, target = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(osp.join(self.img_root, img_file)), cv2.COLOR_BGR2RGB)
        img = self.pipeline(image=img)['image']
        result = {'img': img, 'img_file': img_file, 'target': torch.tensor([target], dtype=torch.float32)}

        return result


@DATASETS.register_module()
class NIHDetectionDataset(Dataset):
    # in xml files: infiltration is named as infiltrate
    nih_bbox_name_to_ind = {
        'Atelectasis': 0,
        'Cardiomegaly': 1,
        'Effusion': 4,
        'Infiltrate': 8,
        'Mass': 9,
        'Nodule': 11,
        'Pneumonia': 13,
        'Pneumothorax': 14
    }

    def __init__(self, img_root: str, annot_root: str, pipeline: List[Dict]) -> None:
        super(NIHDetectionDataset, self).__init__()
        self.img_root = img_root
        self.annot_root = annot_root

        default_args = {'bbox_params': A.BboxParams(format='pascal_voc', label_fields=['labels'])}
        self.pipeline = build_pipeline(pipeline, default_args=default_args)

        self.xml_files = os.listdir(self.annot_root)

        self.ind_to_name = {v: k for k, v in self.nih_bbox_name_to_ind.items()}

    def get_ind_to_name(self) -> Dict[int, str]:
        return self.ind_to_name

    def __len__(self) -> int:
        return len(self.xml_files)

    def __getitem__(self, idx: int) -> Dict:
        # base xml file name, e.g., xxxx_xxxx.xml
        xml_file = self.xml_files[idx]
        img_path = osp.join(self.img_root, osp.splitext(xml_file)[0] + '.png')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bboxes, labels = load_bbox_annot(osp.join(self.annot_root, xml_file), self.nih_bbox_name_to_ind, dtype='float')
        transformed = self.pipeline(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        bboxes = np.asarray(transformed['bboxes'], dtype=int)
        labels = np.asarray(transformed['labels'], dtype=int)
        return {'img_file': osp.basename(img_path), 'img': img, 'bboxes': bboxes, 'labels': labels}


@DATASETS.register_module()
class NIHBinaryDetectionDataset(Dataset):
    # the binary classifier's output has shape (batch_size, 1), and all the pathologies are labeled as positive
    # so map all the pathologies to index 0.
    # nih_bbox_name_to_ind is used for mapping the class labels from xml files to index in the classifier's output
    nih_bbox_name_to_ind = {
        'Atelectasis': 0,
        'Cardiomegaly': 0,
        'Effusion': 0,
        'Infiltrate': 0,
        'Mass': 0,
        'Nodule': 0,
        'Pneumonia': 0,
        'Pneumothorax': 0
    }

    # ind_to_name is used for showing the labels on the top of bboxes.
    ind_to_name = {0: 'Unhealthy'}

    def __init__(self, img_root: str, annot_root: str, pipeline: List[Dict]) -> None:
        super().__init__()
        self.img_root = img_root
        self.annot_root = annot_root

        default_args = {'bbox_params': A.BboxParams(format='pascal_voc', label_fields=['labels'])}
        self.pipeline = build_pipeline(pipeline, default_args=default_args)

        self.xml_files = os.listdir(self.annot_root)

    def get_ind_to_name(self) -> Dict[int, str]:
        return self.ind_to_name

    def __len__(self) -> int:
        return len(self.xml_files)

    def __getitem__(self, idx: int) -> Dict:
        # base xml file name, e.g., xxxx_xxxx.xml
        xml_file = self.xml_files[idx]
        img_path = osp.join(self.img_root, osp.splitext(xml_file)[0] + '.png')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bboxes, labels = load_bbox_annot(osp.join(self.annot_root, xml_file), self.nih_bbox_name_to_ind, dtype='float')
        transformed = self.pipeline(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        bboxes = np.asarray(transformed['bboxes'], dtype=int)
        labels = np.asarray(transformed['labels'], dtype=int)
        return {'img_file': osp.basename(img_path), 'img': img, 'bboxes': bboxes, 'labels': labels}
