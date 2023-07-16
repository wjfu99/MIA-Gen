 # target model
 python training.py \
  --dataset celeba  \
  --model_name vae \
  --model_config './configs/celeba/vae_config.json' \
  --training_config './configs/celeba/base_training_config.json' \
  --train_sta_idx=0 \
  --train_end_idx=120000 \
  --eval_sta_idx=170000 \
  --eval_end_idx=180000

 # target model