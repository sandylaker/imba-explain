data_root = 'data/nih_dataset/'
img_root = data_root + 'test/'
annot_root = data_root + 'bbox_annotations/'

data_type = 'NIHBinaryDetectionDataset'
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
img_size = 224

explain_pipeline = [
    dict(type='Resize', height=img_size, width=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensorV2')
]

data = dict(
    data_loader=dict(batch_size=128, num_workers=4, pin_memory=False, shuffle=False),
    explain=dict(type=data_type, img_root=img_root, annot_root=annot_root, pipeline=explain_pipeline))
