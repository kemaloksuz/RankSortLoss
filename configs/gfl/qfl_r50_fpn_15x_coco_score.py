_base_ = 'qfl_r50_fpn_1x_coco.py'

model = dict(bbox_head=dict(score_weighting=True))

lr_config = dict(step=[12, 16])
total_epochs = 18
