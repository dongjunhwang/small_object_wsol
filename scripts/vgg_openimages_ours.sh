python main.py \
--data_root dataset/ \
--mask_root dataset/ \
--architecture vgg16 \
--wsol_method crop \
--dataset_name OpenImages \
--epochs 10 \
--lr_decay_frequency 3 \
--batch_size 32 \
--proxy_training_set FALSE \
--experiment_name vgg_openimages_ours \
--eval_checkpoint_type last \
--large_feature_map FALSE \
--lr 0.0002402852053 \
--weight_decay 5.00E-04 \
--loss_pos 0.10 \
--loss_neg 0.70 \
--loss_ratio 0.70 \
--crop_threshold 0.05 \
--crop_ratio 0.1
#--checkpoint_path ckpt/.pth.tar \
#--eval_on_val_and_test False \
#--eval_size_ratio True