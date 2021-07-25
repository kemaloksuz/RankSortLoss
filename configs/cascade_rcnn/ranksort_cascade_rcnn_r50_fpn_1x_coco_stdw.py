_base_ = 'ranksort_cascade_rcnn_r50_fpn_1x_coco.py'


model = dict(roi_head=dict(stage_loss_weights=[1, 0.50, 0.25]))
