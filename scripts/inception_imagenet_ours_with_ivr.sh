python main.py \
--data_root dataset/ \
--architecture inception_v3 \
--wsol_method crop \
--dataset_name ILSVRC \
--lr 0.0001260374694 \
--loss_pos 0.30 \
--loss_neg 0.10 \
--loss_ratio 0.60 \
--crop_threshold 0.80 \
--crop_ratio 0.15 \
--epochs 10 \
--lr_decay_frequency 3 \
--batch_size 32 \
--weight_decay 5.00E-04 \
--large_feature_map TRUE \
--proxy_training_set FALSE \
--experiment_name inception_imagenet_ivr \
--eval_checkpoint_type last \
--norm_method ivr \
--percentile 0.40 \
--crop_with_norm TRUE
# --checkpoint_path ckpt/.pth.tar \
# --eval_on_val_and_test False \
# --eval_size_ratio True