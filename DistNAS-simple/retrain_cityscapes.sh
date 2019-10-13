CUDA_VISIBLE_DEVICES=0 python train_new_model.py \
 --batch-size 12 --dataset cityscapes --checkname baseline \
 --epoch 5000 --filter_multiplier 20 --backbone autodeeplab \
 --resize 1024 --crop_size 769 \
 --workers 10 --lr 0.02 \
 --saved-arch-path /home/user/DistNAS-simple/run/cityscapes/beta/experiment_0 \
# --use_amp --opt_level O2 \
# --resume /home/user/DistNAS-simple/run/cityscapes/baseline/experiment_1/checkpoint.pth.tar
