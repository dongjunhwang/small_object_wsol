python main.py \
--data_root dataset/ \
--architecture inception_v3 \
--wsol_method crop \
--dataset_name CUB \
--crop_start_epoch 25 \
--attention_cam TRUE \
--epochs 50 \
--lr_decay_frequency 15 \
--batch_size 32 \
--large_feature_map TRUE \
--proxy_training_set FALSE \
--experiment_name inception_cub_ours \
--eval_checkpoint_type last \
--lr 0.004203227536 \
--weight_decay 5.00E-04 \
--loss_pos 0.90 \
--loss_neg 0.60 \
--loss_ratio 0.60 \
--crop_threshold 0.15 \
--crop_ratio 0.0
#--checkpoint_path ckpt/.pth.tar \
#--eval_on_val_and_test False \
#--eval_size_ratio True \