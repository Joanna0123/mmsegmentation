# optimizer
optimizer = dict(type='AdamW', lr=1e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
