_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py',

]
# optimizer
paths = ['TD', 'TD', 'TD', 'FS', 'none', 'skip_connect', 'BU', 'skip_connect', 'none', 'TD', 'BU', 'none', 'FS', 'SE', 'BU']

stack = 5
find_unused_parameters=True
edge_num = stack * (stack + 1) // 2
model = dict(
    neck=dict(
        type='OPA_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=112,
        num_outs=5,
        paths=paths,
        stack=stack,
        search=False
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=112,
        feat_channels=112,
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        bbox_roi_extractor=dict(
            out_channels=112),
        bbox_head=dict(
            in_channels=112,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                           loss_weight=1.0))
    )
)

