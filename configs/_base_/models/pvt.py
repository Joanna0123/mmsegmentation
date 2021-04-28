# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretained='None',
    backbone=dict(
        type='pvt_tiny_sample_relative_patchembedconv33',
        img_size=769),
    decode_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=3,
        channels=128,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
