_base_ = [
    '../../_base_/classifiers/truncated_densenet.py', '../../_base_/datasets/chexpert_class_dataset.py',
    '../../_base_/losses/bce.py', '../../_base_/schedules/50e.py'
]

data = dict(data_loader=dict(weighted_sampler=True), train=dict(clip_ratio=0.75))
