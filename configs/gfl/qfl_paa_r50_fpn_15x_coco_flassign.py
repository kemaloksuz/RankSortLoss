_base_ = 'fl_paa_r50_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(
        loss_cls=dict(_delete_=True,
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        assign_type = 'qfl'))

lr_config = dict(step=[12, 16])
total_epochs = 18
