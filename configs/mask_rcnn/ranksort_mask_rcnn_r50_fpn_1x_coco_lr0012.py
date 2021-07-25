_base_ = 'ranksort_mask_rcnn_r50_fpn_1x_coco.py'

optimizer = dict(type='SGD', lr=0.012, momentum=0.9, weight_decay=0.0001)
