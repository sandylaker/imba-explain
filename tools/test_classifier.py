import mmcv
import torch
from argparse import ArgumentParser
from mmcv import DictAction

from imba_explain.test_apis import test_classifier


def parse_args():
    parser = ArgumentParser('Test a classifier.')
    parser.add_argument('config', help='Path to the config file.')
    parser.add_argument('ckpt', help='Path to the checkpoint file.')
    parser.add_argument(
        '--work-dir',
        default='workdirs/',
        help="Directory for storing output files. Predictions will be stored under 'work_dir/test_results'.")
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair in xxx=yyy will be merged into config file.'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    mmcv.mkdir_or_exist(args.work_dir)
    cfg.work_dir = args.work_dir

    device = torch.device(f'cuda:{args.gpu_id}')

    test_classifier(cfg, ckpt=args.ckpt, device=device)


if __name__ == '__main__':
    main()
