import os.path as osp

import ignite.distributed as idist
import mmcv
from argparse import ArgumentParser
from mmcv import DictAction

from imba_explain.train_apis import train_classifier


def parser_args():
    parser = ArgumentParser('Train a classifier.')
    parser.add_argument('config', help='Path to the config file.')
    parser.add_argument('--work-dir', default='workdirs', help='Working directory to store the output files.')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--sync-bn', action='store_true', help='Synchronized BatchNorm.')
    parser.add_argument('--backend', help='Distributed backend.')
    parser.add_argument('--nproc_per_node', type=int, help='Number of processes per node.')
    parser.add_argument('--nnodes', type=int, help='Number of nodes.')
    parser.add_argument('--node_rank', type=int, help='Node rank.')
    parser.add_argument('--master_addr', help='Master address.')
    parser.add_argument('--master_port', type=int, help='Master port.')
    parser.add_argument('--init_method', help='Initialization method of distributed training.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair in xxx=yyy will be merged into config file.'
    )

    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.seed is not None:
        cfg.update({'seed': args.seed})
    cfg.update({'sync_bn': args.sync_bn})

    work_dir = args.work_dir
    mmcv.mkdir_or_exist(work_dir)
    cfg.update({'work_dir': work_dir})
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    dist_kwargs = {
        'backend': args.backend,
        'nproc_per_node': args.nproc_per_node,
        'nnodes': args.nnodes,
        'node_rank': args.node_rank,
        'master_addr': args.master_addr,
        'master_port': args.master_port,
        'init_method': args.init_method
    }

    with idist.Parallel(**dist_kwargs) as parallel:
        parallel.run(train_classifier, cfg)


if __name__ == '__main__':
    main()
