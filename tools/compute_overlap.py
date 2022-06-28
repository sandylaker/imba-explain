import os.path as osp
from typing import Dict, List, Optional

import cv2
import mmcv
import numpy as np
from argparse import ArgumentParser
from functools import partial
from glob import glob
from ignite.utils import setup_logger
from tqdm import tqdm

from imba_explain.explain_apis import Overlap


def parse_args():
    parser = ArgumentParser('Run the Intersection over Union.')
    parser.add_argument('attr_map_dir', help='Directory where the attribution maps are stored.')
    parser.add_argument('bboxes_records', help='Json file that records the bounding boxes of attribution maps.')
    parser.add_argument('--mode', choices=['iobb', 'ior', 'iou'], default='iobb', help='Mode for computing overlap.')
    parser.add_argument(
        '--attr-threshold', type=float, help='Attribution value threshold. If None, perform soft IoBB, IoR, or IoU.')

    args = parser.parse_args()
    return args


def run_overlap(attr_map_dir: str,
                bboxes_records: str,
                mode: str = 'iobb',
                attr_threshold: Optional[float] = None) -> None:
    logger = setup_logger('imba-explain')
    logger.info(f'Loading bboxes records from {bboxes_records}.')
    bboxes_records: Dict[str, Dict[str, List]] = mmcv.load(bboxes_records)
    sample_key = next(iter(bboxes_records.keys()))
    if sample_key.endswith('.npy'):
        img_reader_fn = np.load
        file_ext = '.npy'
    else:
        img_reader_fn = partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED)
        file_ext = '.png'

    attr_map_paths = glob(osp.join(attr_map_dir, f'**/*{file_ext}'), recursive=True)
    logger.info(f'There are {len(attr_map_paths)} attribution maps.')

    overlap_evaluator = Overlap(mode=mode, attr_threshold=attr_threshold)

    overlap_list = []
    for attr_map_path in tqdm(attr_map_paths):
        attr_map = img_reader_fn(attr_map_path)
        if file_ext == '.png':
            attr_map = attr_map.astype(float) / 255.0
        base_name = osp.basename(attr_map_path)
        bboxes = np.asarray(bboxes_records[base_name]['bboxes'])
        overlap_result = overlap_evaluator.evaluate(attr_map, bboxes)
        overlap_list.append(overlap_result)

    overlap_mean = np.mean(overlap_list)
    overlap_std = np.std(overlap_list)
    logger.info(f'{mode.swapcase()}: mean: {overlap_mean:.4f}; std: {overlap_std:.4f}')


def main():
    args = parse_args()
    run_overlap(
        attr_map_dir=args.attr_map_dir,
        bboxes_records=args.bboxes_records,
        mode=args.mode,
        attr_threshold=args.attr_threshold)


if __name__ == '__main__':
    main()
