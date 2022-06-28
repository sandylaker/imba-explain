_base_ = [
    '../../_base_/classifiers/resnet50.py', '../../_base_/datasets/nih_class_dataset.py', '../../_base_/losses/bce.py',
    '../../_base_/schedules/50e.py'
]

data = dict(data_loader=dict(weighted_sampler=True), train=dict(clip_ratio=0.75))
