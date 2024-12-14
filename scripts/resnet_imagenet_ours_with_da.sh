python main.py \
--data_root dataset/ \
--architecture resnet50 \
--wsol_method crop \
--dataset_name ILSVRC \
--lr 0.00001305290365 \
--loss_pos 0.80 \
--loss_neg 0.80 \
--loss_ratio 0.20 \
--crop_threshold 0.85 \
--crop_ratio 0.1 \
--train_only_classifier TRUE \
--epochs 10 \
--lr_decay_frequency 3 \
--batch_size 32 \
--weight_decay 1.00E-04 \
--large_feature_map TRUE \
--proxy_training_set TRUE \
--experiment_name resnet_imagenet_da \
--box_v2_metric True \
--eval_checkpoint_type best \
--checkpoint_path ckpt/crop/resnet_imagenet_crop_da.pth.tar
#--eval_on_val_and_test False \
#--eval_size_ratio True