python main.py \
--data_root dataset/ \
--epochs 10 \
--lr_decay_frequency 3 \
--batch_size 32 \
--weight_decay 1.00E-04 \
--large_feature_map TRUE \
--proxy_training_set TRUE \
--experiment_name resnet_imagenet_ivr \
--eval_checkpoint_type last \
--architecture resnet50 \
--wsol_method crop \
--dataset_name ILSVRC \
--lr 0.00003108411 \
--crop_threshold 0.15 \
--loss_ratio 0.90 \
--loss_pos 0.90 \
--loss_neg 0.10 \
--crop_ratio 0.3 \
--norm_method ivr \
--percentile 0.30 \
--crop_with_norm TRUE
#--checkpoint_path /.pth.tar \
#--eval_on_val_and_test False \
#--eval_size_ratio True