_base_ = 'ranksort_cascade_rcnn_r50_fpn_1x_coco.py'

optimizer = dict(type='SGD', lr=0.009, momentum=0.9, weight_decay=0.0001)