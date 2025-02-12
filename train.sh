cd src

# pretrain on simulated data
# python3 train.py experiment=241212-my_sim_241205-classmap_focalTverskyPp_07_20_13-monai_unet_d32_512_res1_head1_bn-s64_128-lr1e-3-grad1-bs4_2_2-ep300-transV1-preV1
# python3 train.py experiment=241224-my_sim_241221-classmap_focalTverskyPp_07_20_13-hengck23_tf_efficientnetv2_b2_d64_256-s64_128-scratch-lr1e-3-grad1-bs4_2_2-ep300-transV3-preV4
# python3 train.py experiment=250109-my_sim_241221-focalTverskyPp-hengck23_v3_cn_nano_m2_d64_256-s64_128-lr1e-3-grad1-bs4_2_2-ep300-transV3-preV4
# python3 train.py experiment=250111-my_sim_241221-focalTverskyPp-monai_segresnet_f16_bn_d1224-s64_128-lr1e-3-grad1-bs4_2_2-ep300-transV3-preV4
# python3 train.py experiment=250112-my_sim_241221-focalTverskyPp-hengck23_resnet34d_d64_256-s64_128-lr1e-3-grad1-bs4_2_2-ep300-transV3-preV4

# CV
# python3 train_cv.py

# train on all data for submission
python3 train_all_data.py experiment=250101-particle_hard_masks_r0.5-focalTverskyPp-pretrained_241221_299-hengck23_tf_efficientnetv2_b2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4
python3 train_all_data.py experiment=250102-hard_r0.5-focalTverskyPp-pretrained_241205_299-monai_unet_d32_512_res1_head1_bn-s64_128-lr1e-3-bs4_2_2-ep100-transV1-preV1
python3 train_all_data.py experiment=250103-particle_hard_masks_r0.5-focalTverskyPp-hengck23_tf_efficientnetv2_b2_d64_256-s64_128-lr1e-3-bs4_2_2-ep100-transV3-preV4
python3 train_all_data.py experiment=250109-hard-focalTverskyPp-pretrained_241221_299-hengck23_v3_cn_nano_m2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4
python3 train_all_data.py experiment=250111-focalTverskyPp-pretrained_241221_299-monai_segresnet_f16_bn_d1224-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV3-preV4
python3 train_all_data.py experiment=250113-focalTverskyPp-hengck23_enb2_d64_256-s64_256-lr1e-3-bs4_2_2-ep100-transV3-preV4
python3 train_all_data.py experiment=250116-focalTverskyPp-pretrained_241221_299-hengck23_resnet34d_d64_256-s64_256-lr1e-3-bs4_2_2-ep50-transV3-preV4
python3 train_all_data.py experiment=250117-focalTverskyPp-pretrained_241221_299-hengck23_resnet34d_d64_256-s64_256-lr1e-3-bs4_2_2-ep80-transV4-preV4
python3 train_all_data.py experiment=250118-focalTverskyPp-hengck23_env2b2_d64_256-s64_128-lr1e-3_decay05-bs4_2_2-mix_sim-ep100-transV4-preV4
python3 train_all_data.py experiment=250118-focalTverskyPp-hengck23_enb2_d64_256-disBA-s64_128-lr1e-3_decay05-bs4_2_2-ep100-transV4-preV4