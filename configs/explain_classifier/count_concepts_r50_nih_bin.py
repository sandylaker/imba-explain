_base_ = [
    '../_base_/datasets/nih_binary_detect_dataset.py', '../_base_/explain/count_concepts.py',
    '../_base_/classifiers/resnet50.py'
]

# Discard train and val datasets
data = dict(data_loader=dict(batch_size=64, shuffle=False))
classifier = dict(num_classes=1)
concept_detector_cfg = dict(quantile_threshold=0.96)
