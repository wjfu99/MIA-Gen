 # target model
 python training.py \
  --dataset celeba  \
  --model_name vae \
  --model_config './configs/celeba/vae_config.json' \
  --training_config './configs/celeba/base_training_config.json' \
  --train_sta_idx=0 \
  --train_end_idx=50000 \
  --eval_sta_idx=50000 \
  --eval_end_idx=60000

 # target model