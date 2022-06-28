_base_ = [
    '../_base_/datasets/nih_binary_detect_dataset.py', '../_base_/explain/gradcam.py',
    '../_base_/classifiers/truncated_densenet.py'
]

classifier = dict(num_classes=1)
attribution_method = dict(target_layer='features.norm5', attr_normalizer=dict(type='MinMaxNormalizer'))
