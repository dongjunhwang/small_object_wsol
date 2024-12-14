python main.py \
--data_root dataset/ \
--architecture resnet50 \
--wsol_method brid \
--dataset_name ILSVRC \
--lr 0.00003397832054 \
--loss_pos 0.60 \
--loss_neg 1.00 \
--loss_ratio 0.70 \
--crop_threshold 0.85 \
--crop_ratio 0.1 \
--iou_threshold_list 30 50 70 \
--pretrained TRUE \
--epochs 10 \
--lr_decay_frequency 3 \
--batch_size 32 \
--crop True \
--weight_decay 1.00E-04 \
--large_feature_map TRUE \
--proxy_training_set TRUE \
--experiment_name resnet_imagenet_brid \
--box_v2_metric True \
--eval_checkpoint_type best \
--checkpoint_path ckpt/resnet_imagenet_brid.pth.tar
#--eval_on_val_and_test False \
#--eval_size_ratio True