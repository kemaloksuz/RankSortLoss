_base_ = 'fl_r50_fpn_1x_coco.py'

model = dict(bbox_head=dict(score_weighting=True))
