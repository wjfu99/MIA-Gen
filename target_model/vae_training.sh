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

 # shadow model
  python training.py \
  --dataset celeba  \
  --model_name vae \
  --model_config './configs/celeba/vae_config.json' \
  --training_config './configs/celeba/base_training_config.json' \
  --train_sta_idx=60000 \
  --train_end_idx=110000 \
  --eval_sta_idx=110000 \
  --eval_end_idx=120000

   # reference model
  python training.py \
  --dataset celeba  \
  --model_name vae \
  --model_config './configs/celeba/vae_config.json' \
  --training_config './configs/celeba/base_training_config.json' \
  --train_sta_idx=120000 \
  --train_end_idx=170000 \
  --eval_sta_idx=170000 \
  --eval_end_idx=180000