python main.py \
--data_root dataset/ \
--mask_root dataset/ \
--architecture resnet50 \
--wsol_method crop \
--dataset_name OpenImages \
--epochs 10 \
--lr_decay_frequency 3 \
--batch_size 32 \
--proxy_training_set FALSE \
--experiment_name resnet_openimages_ours \
--eval_checkpoint_type last \
--large_feature_map FALSE \
--lr 0.0004009288426 \
--weight_decay 1.00E-04 \
--loss_pos 0.70 \
--loss_neg 0.30 \
--loss_ratio 0.30 \
--crop_threshold 0.65 \
--crop_ratio 0
#--checkpoint_path ckpt/.pth.tar \
#--eval_on_val_and_test False \
#--eval_size_ratio True