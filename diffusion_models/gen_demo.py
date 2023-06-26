# !pip install diffusers
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline


model_id = "google/ddpm-cifar10-32"
pipeline = DDPMPipeline.from_pretrained(model_id).to("cuda")
image = pipeline().images[0]
image


# save image
image.save("ddpm_generated_image.png")
