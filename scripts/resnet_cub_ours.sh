python main.py \
--data_root dataset/ \
--architecture resnet50 \
--wsol_method crop \
--dataset_name CUB \
--crop_start_epoch 25 \
--attention_cam TRUE \
--iou_threshold_list 30 50 70 \
--epochs 50 \
--lr_decay_frequency 15 \
--batch_size 16 \
--large_feature_map TRUE \
--proxy_training_set FALSE \
--experiment_name resnet_cub_ours \
--eval_checkpoint_type last \
--lr 0.0005253329993 \
--weight_decay 5.00E-04 \
--loss_pos 1.00 \
--loss_neg 0.10 \
--loss_ratio 0.65 \
--crop_threshold 0.35 \
--crop_ratio 0.4
#--checkpoint_path ckpt/.pth.tar \
#--eval_on_val_and_test False \
#--eval_size_ratio True \