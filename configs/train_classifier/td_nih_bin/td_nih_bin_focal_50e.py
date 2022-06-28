_base_ = [
    '../../_base_/classifiers/truncated_densenet.py', '../../_base_/datasets/nih_binary_class_dataset.py',
    '../../_base_/losses/focal.py', '../../_base_/schedules/50e.py'
]

classifier = dict(num_classes=1)
class_metrics = dict(accuracy=dict(is_multilabel=False))
