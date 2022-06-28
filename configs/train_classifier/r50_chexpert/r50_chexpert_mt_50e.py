_base_ = [
    '../../_base_/classifiers/resnet50.py', '../../_base_/datasets/chexpert_class_dataset.py',
    '../../_base_/losses/multi_task.py', '../../_base_/schedules/50e.py'
]

classifier = dict(num_classes=14)
loss = dict(num_classes=14)
