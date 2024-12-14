python main.py \
--data_root dataset/ \
--architecture vgg16 \
--wsol_method brid \
--dataset_name ILSVRC \
--lr 0.00006680514664 \
--loss_pos 1.00 \
--loss_neg 0.00 \
--loss_ratio 0.40 \
--crop_threshold 0.75 \
--crop_ratio 1 \
--train_only_classifier FALSE \
--pretrained TRUE \
--epochs 10 \
--lr_decay_frequency 3 \
--batch_size 32 \
--crop True \
--weight_decay 5.00E-04 \
--large_feature_map FALSE \
--proxy_training_set TRUE \
--experiment_name vgg_imagenet_brid \
--box_v2_metric True \
--eval_checkpoint_type best \
--checkpoint_path ckpt/vgg_imagenet_brid.pth.tar
#--eval_on_val_and_test False \
#--eval_size_ratio True