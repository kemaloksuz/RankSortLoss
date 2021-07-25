_base_ = '../faster_rcnn/ranksort_faster_rcnn_r50_fpn_1x_coco.py'

optimizer = dict(type='SGD', lr=0.020, momentum=0.9, weight_decay=0.0001)
