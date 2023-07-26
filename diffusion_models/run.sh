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


#accelerate launch training_general.py \
#  --train_data_dir="/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total" \
#  --resume_from_checkpoint "latest" \
#  --resolution=64 --center_crop \
#  --output_dir="ddpm-celeba-64-target2" \
#  --train_batch_size=16 \
#  --num_epochs=500 \
#  --gradient_accumulation_steps=1 \
#  --learning_rate=1e-4 \
#  --lr_warmup_steps=500 \
#  --mixed_precision=no \
#  --train_sta_idx=0 \
#  --train_end_idx=10000 \
#  --eval_sta_idx=10000 \
#  --eval_end_idx=11000
# for 100k training datasets
#accelerate launch training_general.py \
#  --train_data_dir="/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total" \
#  --resume_from_checkpoint "latest" \
#  --resolution=64 --center_crop \
#  --output_dir="ddpm-celeba-64-100k" \
#  --train_batch_size=16 \
#  --num_epochs=200 \
#  --checkpointing_steps=1500 \
#  --gradient_accumulation_steps=1 \
#  --learning_rate=5e-11 \
#  --lr_warmup_steps=500 \
#  --mixed_precision=no \
#  --train_sta_idx=0 \
#  --train_end_idx=100000 \
#  --eval_sta_idx=100000 \
#  --eval_end_idx=110000

  # for 50k training datasets target
  accelerate launch training_general.py \
  --train_data_dir="/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total" \
  --resume_from_checkpoint "latest" \
  --resolution=64 --center_crop \
  --output_dir="ddpm-celeba-64-50k" \
  --train_batch_size=16 \
  --num_epochs=400 \
  --checkpointing_steps=1500 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --train_sta_idx=0 \
  --train_end_idx=50000 \
  --eval_sta_idx=50000 \
  --eval_end_idx=60000

    # for 50k training datasets shadow
  accelerate launch training_general.py \
  --train_data_dir="/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total" \
  --resume_from_checkpoint "latest" \
  --resolution=64 --center_crop \
  --output_dir="ddpm-celeba-64-50k-shadow" \
  --train_batch_size=16 \
  --num_epochs=400 \
  --checkpointing_steps=1500 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --train_sta_idx=60000 \
  --train_end_idx=110000 \
  --eval_sta_idx=110000 \
  --eval_end_idx=120000

      # for 50k training datasets reference
  accelerate launch training_general.py \
  --train_data_dir="/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total" \
  --resume_from_checkpoint "latest" \
  --resolution=64 --center_crop \
  --output_dir="ddpm-celeba-64-50k-reference" \
  --train_batch_size=16 \
  --num_epochs=400 \
  --checkpointing_steps=1500 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --train_sta_idx=120000 \
  --train_end_idx=170000 \
  --eval_sta_idx=170000 \
  --eval_end_idx=180000


  #### train ddpm for tiny-in dataset
    accelerate launch training_general.py \
  --dataset_name="Maysee/tiny-imagenet" \
  --resume_from_checkpoint "latest" \
  --resolution=64 --center_crop \
  --output_dir="ddpm-tinyin-64-50k" \
  --train_batch_size=16 \
  --num_epochs=400 \
  --checkpointing_steps=1500 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --train_sta_idx=0 \
  --train_end_idx=30000 \
  --eval_sta_idx=30000 \
  --eval_end_idx=35000

  #### train ddpm for tiny-in dataset
    accelerate launch training_general.py \
  --dataset_name="Maysee/tiny-imagenet" \
  --resume_from_checkpoint "latest" \
  --resolution=64 --center_crop \
  --output_dir="ddpm-tinyin-64-50k-shadow" \
  --train_batch_size=16 \
  --num_epochs=400 \
  --checkpointing_steps=1500 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --train_sta_idx=35000 \
  --train_end_idx=65000 \
  --eval_sta_idx=65000 \
  --eval_end_idx=70000

    #### train ddpm for tiny-in dataset
    accelerate launch training_general.py \
  --dataset_name="Maysee/tiny-imagenet" \
  --resume_from_checkpoint "latest" \
  --resolution=64 --center_crop \
  --output_dir="ddpm-tinyin-64-50k-reference" \
  --train_batch_size=16 \
  --num_epochs=400 \
  --checkpointing_steps=1500 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --train_sta_idx=70000 \
  --train_end_idx=100000 \
  --eval_sta_idx=100000 \
  --eval_end_idx=105000