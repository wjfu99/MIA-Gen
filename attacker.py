import os
import pythae
from pythae.models import AutoModel
from pythae.samplers import NormalSampler

last_training = sorted(os.listdir('my_model'))[-1]
trained_model = AutoModel.load_from_folder(os.path.join('my_model', last_training, 'final_model'))

# create normal sampler
normal_samper = NormalSampler(
    model=trained_model
)

# sample
gen_data = normal_samper.sample(
    num_samples=25,
    output_dir='./data/mnist/gen_data',
    save_sampler_config=True
)