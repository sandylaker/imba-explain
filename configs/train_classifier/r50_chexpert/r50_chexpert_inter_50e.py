_base_ = [
    '../../_base_/classifiers/resnet50.py', '../../_base_/datasets/chexpert_class_dataset.py',
    '../../_base_/losses/inter_wbce.py', '../../_base_/schedules/50e.py'
]

classifier = dict(num_classes=14)
