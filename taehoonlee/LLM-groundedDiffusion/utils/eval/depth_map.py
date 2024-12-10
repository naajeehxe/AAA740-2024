import os
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

input_folder = "/home/taehoonlee/AAA740-2024/taehoonlee/LLM-groundedDiffusion/intermediate_results"
output_folder = os.path.join(input_folder, "midas_result")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        output_img_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_depth.png")
        plt.imsave(output_img_path, output, cmap='inferno')


fig, axes = plt.subplots(len([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]), 2, figsize=(10, 5 * len([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])))

for idx, filename in tqdm(enumerate(sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]))):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    depth_img_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_depth.png")
    depth_img = plt.imread(depth_img_path)

    axes[idx, 0].imshow(img)
    axes[idx, 0].set_title(f"Original Image: {filename}")
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(depth_img, cmap='inferno')
    axes[idx, 1].set_title(f"Depth Map: {filename}")
    axes[idx, 1].axis('off')

plt.tight_layout()
output_plot_path = os.path.join(output_folder, "depth_map_comparison.png")
plt.savefig(output_plot_path)
plt.close()