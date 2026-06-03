from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime

# Create result folder
os.makedirs("result", exist_ok=True)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

pipe = pipe.to(device)

# User input
prompt = input("Enter image description: ")

print("Generating image... Please wait...")

# Generate image
image = pipe(prompt).images[0]

# Unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

file_path = f"result/generated_{timestamp}.png"
plt.imshow(image)
plt.axis("off")
plt.show()
# Save image
image.save(file_path)

print(f"Image saved successfully!")
print(f"Location: {file_path}")