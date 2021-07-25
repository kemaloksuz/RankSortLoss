_base_ = 'ranksort_cascade_rcnn_r50_fpn_1x_coco.py'

model = dict(rpn_head=dict(head_weight=0.20))

optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001)