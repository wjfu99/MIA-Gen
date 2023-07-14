#accelerate launch training_general.py \
#  --train_data_dir="/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total" \
#  --resume_from_checkpoint "latest" \
#  --resolution=64 --center_crop \
#  --output_dir="ddpm-celeba-64-test" \
#  --train_batch_size=16 \
#  --num_epochs=500 \
#  --gradient_accumulation_steps=1 \
#  --learning_rate=1e-4 \
#  --lr_warmup_steps=500 \
#  --mixed_precision=no \
#  --train_sta_idx=150000 \
#  --train_end_idx=160000 \
#  --eval_sta_idx=160000 \
#  --eval_end_idx=170000


accelerate launch training_general.py \
  --train_data_dir="/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total" \
  --resume_from_checkpoint "latest" \
  --resolution=64 --center_crop \
  --output_dir="ddpm-celeba-64-target2" \
  --train_batch_size=16 \
  --num_epochs=500 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --train_sta_idx=0 \
  --train_end_idx=10000 \
  --eval_sta_idx=10000 \
  --eval_end_idx=11000
accelerate launch training_general.py \
  --train_data_dir="/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total" \
  --resume_from_checkpoint "latest" \
  --resolution=64 --center_crop \
  --output_dir="ddpm-celeba-64-100k" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --train_sta_idx=0 \
  --train_end_idx=100000 \
  --eval_sta_idx=100000 \
  --eval_end_idx=110000