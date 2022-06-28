_base_ = ['./draw_bboxes.py']

attribution_method = dict(
    type='GradCAM', target_layer='layer4.2', relu_attributions=True, attr_normalizer=dict(type='IdentityNormalizer'))
