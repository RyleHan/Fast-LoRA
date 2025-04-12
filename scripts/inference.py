# inference.py
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch

def generate(prompt):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    # 加载 LoRA 权重
    pipe.unet = PeftModel.from_pretrained(pipe.unet, "./lora_weights")

    image = pipe(prompt, guidance_scale=7.5).images[0]
    image.save("after_lora.png")
    print("Image saved as after_lora.png")

if __name__ == "__main__":
    generate("a girl standing in the forest, anime sketch style")

