#------------------------------------
#Data:2023/3/22 16:00
#name:gsl
#describe:uda 2teacher_net idea
#------------------------------------
import math

_base_ =[
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_cityscapes_to_medium_acdc_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/TA_multiTeacher.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Psueudo-Label Crop
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    use_ref_label=True)
data = dict(
    train =dict(
        #Rare Class Sampling
        rare_class_sampling=dict(
           min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))#class_temp=0.01
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
# Meta Information for Result Analysis
name = 'cs2acdc_400_uda_2teacher_rcs_croppl_a999_daformer_mitb5_s0'
exp = 'basic'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
