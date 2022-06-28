import os.path as osp
from typing import Dict, List

import cv2
import mmcv
import numpy as np
from argparse import ArgumentParser
from functools import partial
from glob import glob
from ignite.utils import setup_logger
from tqdm import tqdm

from imba_explain.explain_apis import PointingGame


def parse_args():
    parser = ArgumentParser('Run the pointing game.')
    parser.add_argument('attr_map_dir', help='Directory where the attribution maps are stored.')
    parser.add_argument('bboxes_records', help='Json file that records the bounding boxes of attribution maps.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=1,
        help='Top K parameter in the pointing game. By default, only the top-1 pixel counts.')

    args = parser.parse_args()
    return args


def run_pointing_game(attr_map_dir: str, bboxes_records: str, top_k: int = 1) -> None:
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

    pg = PointingGame(top_k=top_k)

    num_true_pos_list = []
    num_pos_list = []
    for attr_map_path in tqdm(attr_map_paths):
        attr_map = img_reader_fn(attr_map_path)
        base_name = osp.basename(attr_map_path)
        bboxes = np.asarray(bboxes_records[base_name]['bboxes'])
        res = pg.evaluate(attr_map, bboxes)
        num_true_pos_list.append(res['num_true_pos'])
        num_pos_list.append(res['num_pos'])

    num_true_pos = np.sum(num_true_pos_list)
    num_pos = np.sum(num_pos_list)
    performance = num_true_pos / (num_pos + 1e-8)
    logger.info(f'Pointing game: True positive: {num_true_pos}; Positive: {num_pos}; Performance: {performance:.4f}')


def main():
    args = parse_args()
    run_pointing_game(attr_map_dir=args.attr_map_dir, bboxes_records=args.bboxes_records, top_k=args.top_k)


if __name__ == '__main__':
    main()
