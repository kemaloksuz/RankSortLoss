_base_ = 'nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py'

model = dict(
    bbox_head=dict(_delete_=True,
        type='RankBasedNASFCOSHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_cfg=dict(type='GN', num_groups=32),
        loss_bbox=dict(type='GIoULoss', reduction='none')))