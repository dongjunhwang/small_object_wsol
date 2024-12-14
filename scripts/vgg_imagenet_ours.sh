python main.py \
--data_root dataset/ \
--architecture vgg16 \
--wsol_method crop \
--dataset_name ILSVRC \
--lr 0.00006680514664 \
--loss_pos 1.00 \
--loss_neg 0.00 \
--loss_ratio 0.40 \
--crop_threshold 0.75 \
--crop_ratio 0.7 \
--epochs 10  \
--lr_decay_frequency 3 \
--batch_size 32 \
--weight_decay 5.00E-04 \
--large_feature_map FALSE \
--proxy_training_set TRUE \
--experiment_name vgg_imagenet_ours \
--num_val_sample_per_class 0 \
--eval_checkpoint_type last
#--checkpoint_path ckpt/.pth.tar
#--eval_on_val_and_test False
#--eval_size_ratio True