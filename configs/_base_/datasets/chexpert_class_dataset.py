data_root = 'data/chexpert_dataset/'
data_type = 'CheXpertDataset'
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
img_size = 224

train_pipeline = [
    dict(type='LongestMaxSize', max_size=280),
    dict(type='RandomResizedCrop', height=img_size, width=img_size),
    dict(type='Transpose', p=0.5),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='ShiftScaleRotate', p=0.5),
    dict(type='RandomBrightnessContrast', brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CoarseDropout', max_height=20, max_width=20, min_holes=2, p=0.5),
    dict(type='ToTensorV2')
]

test_pipeline = [
    dict(type='Resize', height=img_size, width=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensorV2')
]

data = dict(
    data_loader=dict(batch_size=512, shuffle=True, num_workers=16, pin_memory=False),
    train=dict(
        type=data_type,
        data_root=data_root,
        label_csv=data_root + 'train.csv',
        indices_file=data_root + 'train_inds.txt',
        pipeline=train_pipeline),
    val=dict(
        type=data_type,
        data_root=data_root,
        label_csv=data_root + 'train.csv',
        indices_file=data_root + 'val_inds.txt',
        pipeline=test_pipeline),
    test=dict(
        type=data_type,
        data_root=data_root,
        label_csv=data_root + 'train.csv',
        indices_file=data_root + 'test_inds.txt',
        pipeline=test_pipeline))
