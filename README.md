# Are overfitting-free generative models vulnerable to membership inference attack?

This is the official implementation of the paper "Are overfitting-free generative models vulnerable to membership inference attack?".
The proposed **P**robabilistic **F**luctuation **A**ssessing **M**embership **I**nference (PFAMI) is implemented as follows.

![The overall architecture of _PFAMI_](./Framework.png)

## Requirements

- torch>=1.11.0
- pythae>=0.1.1
- diffusers>=0.18.0
- accelerate==0.20.3
- datasets>=2.13.1
- torchvision=>0.12.0
- numpy>=1.23.4
- scikit-learn>=1.1.3
- pyyaml>=6.0
- tqdm>=4.64.1

Dependency can be installed with the following command:

```bash
pip install -r requirements.txt
```


## Target Models Training
  All Diffusion models are built on top of [diffuser](https://huggingface.co/docs/diffusers/index), 
  a go-to library for state-of-the-art diffusion models, 
  on which you can train arbitrary state-of-the-art diffusion models you want. 
  Similarly, all VAEs are deployed by [pythae](https://github.com/clementchadebec/benchmark_VAE), 
  another generative model library with massive VAEs from previous to recent ones.. 
  So you can evaluate our attack algorithm on more diverse generative models, which is what we hope to see.


* ### Diffusion Models
  We recommend to train diffusion models with multi-GPU and [accelerate](https://huggingface.co/docs/accelerate/index), 
  a library that enables the same PyTorch code to be run across any distributed configuration. 
  Below is a sample to train a DDPM on Celeba-64, and the training script for all other diffusion models can be found in the [path](./diffusion_models) folder:
  ```bash
    accelerate launch training_general.py \
    --train_data_dir="Replace with your dataset here" \
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
  ```

* ### VAEs
  Below is a sample to train a vanilla VAE on Celeba-64, and the training script for all other VAE models can be found in the [path](./diffusion_models) folder:
    ```bash
   python training.py \
    --dataset celeba  \
    --model_name vae \
    --model_config './configs/celeba/vae_config.json' \
    --training_config './configs/celeba/base_training_config.json' \
    --train_sta_idx=0 \
    --train_end_idx=50000 \
    --eval_sta_idx=50000 \
    --eval_end_idx=60000
    ```
## Pre-trained model
Pre-trained models can be downloaded from Hugging-face, we will release links after reviewing for anonymization.
  
## Run PFAMI

To execute PFAMI on diffusion models and VAEs, please manually modify the _"target_model"_ item in the config file `config.json`:
* ### Diffusion models

    ```yaml
      random_seed: 42
      target_model: diffusion # valid model: diffusion, vae
      dataset: celeba # valid dataset: celeba, tinyin
      attack_kind: stat # valid attacks: nn, stat
      loss_kind: ddpm # valid loss estimation methods: ddpm, ddim
      time_step: 10
      time_sec: 100
      calibration: true # whether to enable calibration
      sample_number: 1000 # the number of samples for each data group
      eval_batch_size: 100 # batch size of the evaluation phase
      diffusion_sample_number: 10 # the number of equidistant sampling
      diffusion_sample_steps: [0, 50, 100, 150, 200, 250, 300, 350, 400, 450] # the sample steps of diffusion model
      perturbation_number: 10 # the number of query
      extensive_per_num: 10 # sample number, should be set 1 for diffusion models
      start_strength: 0.95 # start strength factor of the perturbation mechanism
      end_strength: 0.7 # end strength factor of the perturbation mechanism
      attack_data_path: attack
      epoch_number: 1000
      load_trained: true # whether to load existing trained attack model
      load_attack_data: true # whether to load prepared attack data if existing.
    ```

* ### VAEs

    ```yaml
      random_seed: 42
      target_model: vae # valid model: diffusion, vae
      dataset: celeba # valid dataset: celeba, tinyin
      attack_kind: stat # valid attacks: nn, stat
      loss_kind: ddpm # valid loss estimation methods: ddpm, ddim
      time_step: 10
      time_sec: 100
      calibration: true # whether to enable calibration
      sample_number: 1000 # the number of samples for each data group
      eval_batch_size: 100 # batch size of the evaluation phase
      diffusion_sample_number: 10 # the number of equidistant sampling
      diffusion_sample_steps: [0, 50, 100, 150, 200, 250, 300, 350, 400, 450] # the sample steps of diffusion model
      perturbation_number: 10 # the number of query
      extensive_per_num: 10 # sample number, should be set 1 for diffusion models
      start_strength: 0.95 # start strength factor of the perturbation mechanism
      end_strength: 0.7 # end strength factor of the perturbation mechanism
      attack_data_path: attack
      epoch_number: 1000
      load_trained: true # whether to load existing trained attack model
      load_attack_data: true # whether to load prepared attack data if existing.
    ```