CUDA_VISIBLE_DEVICES=1 python train_new_model.py \
 --batch-size 16 --dataset cityscapes --checkname retrain \
 --epoch 4500 --filter_multiplier 20 \
 --resize 1024 --crop_size 769