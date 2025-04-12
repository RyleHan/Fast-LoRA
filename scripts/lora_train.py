# lora_train.py
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import torchvision.transforms as T

class SimpleImageDataset(Dataset):
    def __init__(self, image_dir):
        self.paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)

def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["attn2.to_q", "attn2.to_v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    pipe.unet = get_peft_model(pipe.unet, peft_config)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Load dataset
    dataset = SimpleImageDataset("./data/anime_style")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-4)

    pipe.unet.train()
    for epoch in range(3):
        for i, img in enumerate(dataloader):
            img = img.to("cuda")
            loss = pipe(img, guidance_scale=7.5).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

    pipe.unet.save_pretrained("./lora_weights")

if __name__ == "__main__":
    main()

