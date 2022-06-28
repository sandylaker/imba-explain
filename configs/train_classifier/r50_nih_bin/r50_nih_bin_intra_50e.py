_base_ = [
    '../../_base_/classifiers/resnet50.py', '../../_base_/datasets/nih_binary_class_dataset.py',
    '../../_base_/losses/intra_wbce.py', '../../_base_/schedules/50e.py'
]

classifier = dict(num_classes=1)
class_metrics = dict(accuracy=dict(is_multilabel=False))
