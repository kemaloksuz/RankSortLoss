_base_ = 'mask_rcnn_r50_fpn_1x_coco.py'

model = dict(
    rpn_head=dict(
        type='RankBasedRPNHead',
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_bbox=dict(type='GIoULoss', reduction='none'),
        head_weight=0.20),
    roi_head=dict(
        type='RankBasedStandardRoIHead',
        bbox_head=dict(
            type='RankBasedShared2FCBBoxHead',
            reg_decoded_bbox= True,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_bbox=dict(type='GIoULoss', reduction='none'),
            loss_cls=dict(use_sigmoid=True)),
        mask_head=dict(type='RankBasedFCNMaskHead')))
# model training and testing settings
train_cfg = dict(
    rpn=dict(sampler=dict(type='PseudoSampler')),
    rcnn=dict(sampler=dict(num=1e10)))

checkpoint_config = dict(interval=6)

optimizer = dict(type='SGD', lr=0.012, momentum=0.9, weight_decay=0.0001)
