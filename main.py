import os
from dotenv import load_dotenv
import torch
import matplotlib.pyplot as plt

from UNet.train import train_model
from matplotlib.patches import Polygon as MplPolygon
from loadData import SpaceNet7Buildings
from UNet.preprocessChips import ChipsDataset, create_all_chips
from UNet.UNet import UNet

if __name__ == "__main__":
    load_dotenv()  # load variables from .env

    root = os.getenv("SPACENET_ROOT")
    if root is None:
        raise ValueError("SPACENET_ROOT not set in .env!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    print("Loading dataset...")
    sn7_dataset = SpaceNet7Buildings(root, return_series=True)

    print("Generating chips from full dataset...")
    chips = create_all_chips(sn7_dataset, chip_size=256)
    chips_dataset = ChipsDataset(chips)

    # 4 Channels: RGB + NIR
    UNet_Model = UNet(in_channels=4, out_channels=1 )

    print(f"Training on {len(chips_dataset)} chips...")

    # --- Verification: Plot one full image with polygons ---
    sample_tile = sn7_dataset[0]  # pick first tile
    frames = sample_tile.get("frames", [sample_tile])
    sample_frame = frames[0]

    img = sample_frame["image"].permute(1,2,0).numpy()  # HWC
    polygons = sample_frame["polygons_pix"]

    plt.figure(figsize=(8,8))
    plt.imshow(img)
    ax = plt.gca()
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.add_patch(MplPolygon(list(zip(x, y)), closed=True, fill=False, edgecolor='red', linewidth=2))
    plt.title("Full tile image with polygons from SpaceNet7")
    plt.axis('off')
    plt.show()

    # --- Verification: Plot one chip and its mask ---
    chip_img_dict = chips_dataset[0]  # first chip
    chip_img = chip_img_dict["image"].permute(1,2,0).numpy()
    chip_mask = chip_img_dict["mask"][0].numpy()  # (1,H,W) → (H,W)

    plt.figure(figsize=(6,6))
    plt.imshow(chip_img)
    plt.imshow(chip_mask, alpha=0.5, cmap='Reds')
    plt.title("Chip image with mask overlay")
    plt.axis('off')
    plt.show()

    train_model(
        dataset=chips_dataset,
        model=UNet_Model,
        epochs=10,
        batch_size=4,
        lr=1e-4,
        device=device,
        pos_weight=5,
        patience=5
    )