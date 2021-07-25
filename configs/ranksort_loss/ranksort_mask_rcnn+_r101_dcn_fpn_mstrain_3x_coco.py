_base_ = 'ranksort_mask_rcnn+_r101_fpn_mstrain_3x_coco.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))