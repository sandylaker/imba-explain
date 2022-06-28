_base_ = [
    '../_base_/datasets/nih_binary_detect_dataset.py', '../_base_/explain/count_concepts.py',
    '../_base_/classifiers/truncated_densenet.py'
]

# Discard train and val datasets
data = dict(data_loader=dict(batch_size=128, shuffle=False))

classifier = dict(num_classes=1)
target_layer = 'features.norm5'
