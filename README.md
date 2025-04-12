# Fast-LoRA-StableDiffusion


# README.md

# Fast-LoRA-StableDiffusion

> Fast LoRA fine-tuning of Stable Diffusion using only 5 training images to perform lightweight style transfer (e.g. anime sketch, pencil drawing, etc.).

## 🚀 Features

- LoRA fine-tuning with [PEFT](https://github.com/huggingface/peft) for memory-efficient training
- Minimal GPU memory usage (<6GB)
- Quick training (under 1 hour on a consumer GPU)
- Style transfer on Stable Diffusion with only 5 images
- Easy to extend for other styles or domains

## 🧠 Tech Stack

- Stable Diffusion v1.5 ([runwayml](https://huggingface.co/runwayml/stable-diffusion-v1-5))
- Diffusers + PEFT
- LoRA for efficient training

## 📁 Project Structure

```
Fast-LoRA-StableDiffusion/
├── lora_train.py              # LoRA training script
├── inference.py              # Inference with LoRA weights
├── configs/                  # Config files (optional)
├── images/                   # Sample images (before/after)
├── data/
│   └── anime_style/          # Training images (5~10 JPEGs)
├── README.md
└── requirements.txt
```

## 🏃 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
Put 5~10 style reference images in `./data/anime_style`.

### 3. Train LoRA
```bash
python lora_train.py
```

### 4. Run Inference
```bash
python inference.py
```
Output saved as `after_lora.png`

## 🖼️ Example

| Original | Stylized with LoRA |
|----------|--------------------|
| ![](images/before.png) | ![](images/after_lora.png) |

## 📄 License
MIT
