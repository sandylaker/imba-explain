_base_ = ['./draw_bboxes.py']

attribution_method = dict(type='ExtremalPerturbation', areas=[0.1], attr_normalizer=dict(type='IdentityNormalizer'))
