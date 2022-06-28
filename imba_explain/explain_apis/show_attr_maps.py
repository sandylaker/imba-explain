import os.path as osp
from copy import deepcopy
from typing import Union

import cv2
import mmcv
import numpy as np
import torch
from ignite.utils import manual_seed, setup_logger
from mmcv import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..classifiers import build_classifier
from ..datasets import bbox_collate_fn, build_dataset
from .bbox_visualizer import add_multiple_labels, draw_multiple_rectangles
from .builder import build_attributions


def show_attr_maps(cfg: Config,
                   ckpt: str,
                   as_npy: bool = False,
                   with_color: bool = False,
                   plot_bboxes: bool = True,
                   save_bboxes: bool = False,
                   single_folder: bool = True,
                   device: Union[str, torch.device] = 'cuda:0',
                   with_pbar: bool = True) -> None:
    manual_seed(cfg.get('seed', 2022))
    logger = setup_logger('imba-explain')
    if as_npy:
        if with_color or plot_bboxes:
            logger.info("When 'as_npy' is True, 'with_color' and 'plot_bboxes' are forced to be False.")
            with_color = False
            plot_bboxes = False

    if plot_bboxes:
        if not with_color:
            logger.info("When 'plot_bboxes' is True, 'with_color' is forced to be True.")
            with_color = True

    explain_set = build_dataset(cfg.data['explain'])
    logger.info(f'Dataset under: {explain_set.img_root} contains {len(explain_set)} images')
    # a dict that maps label indices to bbox label names
    ind_to_name = explain_set.get_ind_to_name()
    if not single_folder:
        for name in ind_to_name.values():
            mmcv.mkdir_or_exist(osp.join(cfg.work_dir, name))
    data_loader_cfg = deepcopy(cfg.data['data_loader'])
    data_loader_cfg.update({'shuffle': False, 'drop_last': False})
    logger.info(f'Dataloader config: {data_loader_cfg}')
    explain_loader = DataLoader(explain_set, collate_fn=bbox_collate_fn, **data_loader_cfg)

    state_dict = torch.load(ckpt, map_location='cpu')
    logger.info(f'Using the checkpoint: {ckpt}')
    classifier = build_classifier(cfg.classifier)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    attribution_method = build_attributions(cfg.attribution_method)
    attribution_method.set_classifier(classifier)

    pbar = tqdm(total=len(explain_set)) if with_pbar else None
    bboxes_records = dict() if save_bboxes else None

    for batch in explain_loader:
        imgs = batch['img'].to(device)
        img_files = batch['img_file']
        bboxes_batch = batch['bboxes']
        labels_batch = batch['labels']

        for img, img_file, bboxes, labels in zip(imgs, img_files, bboxes_batch, labels_batch):
            img = img.unsqueeze(0)
            height, width = img.shape[2], img.shape[3]
            bboxes = bboxes.astype(int)
            labels = labels.astype(int)
            if labels.size == 0:
                continue

            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                attr_map = attribution_method(img, int(label), convert_to_img=not as_npy)
                attr_map = cv2.resize(attr_map, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
                if with_color:
                    # attr_map is in BGR mode.
                    attr_map = cv2.applyColorMap(attr_map, cv2.COLORMAP_VIRIDIS)

                bboxes_to_draw = bboxes[labels == label]
                labels_to_draw = labels[labels == label]
                if plot_bboxes:
                    label_texts = [f'{ind_to_name[i]}' for i in labels_to_draw]
                    attr_map = draw_multiple_rectangles(attr_map, bboxes_to_draw, **cfg.bbox_cfg)
                    attr_map = add_multiple_labels(attr_map, label_texts, bboxes_to_draw, **cfg.bbox_label_cfg)

                img_name = osp.splitext(osp.basename(img_file))[0]
                suffix = '' if i == 0 else f'-{i + 1}'
                file_ext = 'npy' if as_npy else 'png'
                out_base_name = f'{img_name}{suffix}.{file_ext}'
                out_path = osp.join(cfg.work_dir, out_base_name) if single_folder else osp.join(
                    cfg.work_dir, ind_to_name[label], out_base_name)

                if as_npy:
                    np.save(out_path, attr_map)
                else:
                    cv2.imwrite(out_path, attr_map)

                if save_bboxes:
                    to_record = {'bboxes': bboxes_to_draw.tolist(), 'labels': labels_to_draw.tolist()}
                    bboxes_records.update({out_base_name: to_record})

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    if save_bboxes:
        out_path = osp.join(cfg.work_dir, 'bboxes_records.json')
        mmcv.dump(bboxes_records, out_path)
        logger.info(f'Bboxes records have been saved to {out_path}.')
