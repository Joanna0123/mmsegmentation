# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
<<<<<<< HEAD
    pretrained='/home/ma-user/work/pspnet/mmsegmentation/configs/fcn/pvt_tiny_sample_relative_patchembedconv33_model.pth',
=======
    pretained='../../../mmseg/models/backbones/pvt_tiny_sample_relative_patchembedconv33_model.pth',
>>>>>>> d9e8565d35f8abf22e5b70ce19e0ff161ab6d638
    backbone=dict(
        type='pvt_tiny_sample_relative_patchembedconv33',
        img_size=769),
    decode_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=0,
        channels=256,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(769,769), stride=(513,513)))
