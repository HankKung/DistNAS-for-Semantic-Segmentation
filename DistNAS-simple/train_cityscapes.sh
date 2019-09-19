CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py \
 --batch-size 4 --dataset cityscapes --checkname dist \
 --alpha_epoch 25 --epoch 50 --filter_multiplier 8 \
 --resize 512 --crop_size 321