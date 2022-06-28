import mmcv
import torch
from argparse import ArgumentParser
from mmcv import DictAction

from imba_explain.explain_apis import show_attr_maps


def parse_args():
    parser = ArgumentParser('Show attribution maps.')
    parser.add_argument('config', help='Path to a configuration file.')
    parser.add_argument('ckpt', help='Path to a checkpoint file.')
    parser.add_argument('--work-dir', default='workdirs/', help='Directory in which the output files will be stored.')
    parser.add_argument(
        '--as-npy',
        action='store_true',
        help='Whether to disable converting the attribution maps to images. If True, store the'
        'attribution maps as npy files, and force with_color and plot_bboxes to False.')
    parser.add_argument(
        '--with-color',
        action='store_true',
        help='Whether to apply colormap. When plot_bboxes is True, this argument is forced to be True.')
    parser.add_argument(
        '--save-bboxes',
        action='store_true',
        help='Whether save the bounding boxes of attribution maps to a json file.')
    parser.add_argument(
        '--plot-bboxes', action='store_true', help='Whether to plot bounding boxes on the attribution maps.')
    parser.add_argument(
        '--single-folder', action='store_true', help='Whether to save all the attribution maps in a single folder.')
    parser.add_argument('--with-pbar', action='store_true', help='Whether to use a progress bar.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument('--seed', type=int, help='Random seed.')

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
    show_attr_maps(
        cfg,
        ckpt=args.ckpt,
        as_npy=args.as_npy,
        with_color=args.with_color,
        save_bboxes=args.save_bboxes,
        plot_bboxes=args.plot_bboxes,
        single_folder=args.single_folder,
        device=device,
        with_pbar=args.with_pbar)


if __name__ == '__main__':
    main()
