python main.py \
--data_root dataset/ \
--mask_root dataset/ \
--architecture inception_v3 \
--wsol_method crop \
--dataset_name OpenImages \
--epochs 10 \
--lr_decay_frequency 3 \
--batch_size 32 \
--proxy_training_set FALSE \
--experiment_name inception_openimages_ours \
--eval_checkpoint_type last \
--large_feature_map True \
--lr 0.000148741774 \
--weight_decay 5.00E-04 \
--loss_pos 0.20 \
--loss_neg 1.30 \
--loss_ratio 0.90 \
--crop_threshold 0.15 \
--crop_ratio 0.1
#--checkpoint_path ckpt/.pth.tar \
#--eval_on_val_and_test False \
#--eval_size_ratio True