_base_ = [
    '../../_base_/classifiers/truncated_densenet.py', '../../_base_/datasets/chexpert_class_dataset.py',
    '../../_base_/losses/bce.py', '../../_base_/schedules/50e.py'
]

classifier = dict(num_classes=14)
