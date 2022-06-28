_base_ = [
    '../_base_/datasets/nih_detect_dataset.py', '../_base_/explain/gradcam.py', '../_base_/classifiers/resnet50.py'
]

attribution_method = dict(attr_normalizer=dict(type='MinMaxNormalizer'))
