import os
from dotenv import load_dotenv
import torch
import matplotlib.pyplot as plt

from UNet.train import train_model
from matplotlib.patches import Polygon as MplPolygon
from loadData import SpaceNet7Buildings
from UNet.chips import ChipsDataset, create_all_chips
from UNet.UNet import UNet
from UNet.evaluate import evaluate_model

def save_verification_plots(sn7_dataset, chips_dataset, save_dir="verification_plots"):
    """
    Saves:
      - full tile image with polygons
      - one chip image with mask overlay
    """
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. FULL TILE + POLYGON OVERLAY
    # --------------------------------------------------
    sample_tile = sn7_dataset[0]              # pick first tile
    frames = sample_tile.get("frames", [sample_tile])
    sample_frame = frames[0]

    img = sample_frame["image"].permute(1, 2, 0).numpy()   # HWC
    polygons = sample_frame["polygons_pix"]

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    ax = plt.gca()
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.add_patch(MplPolygon(list(zip(x, y)), closed=True, fill=False,
                                edgecolor='red', linewidth=2))
    plt.title("Full Tile with Polygons")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tile_polygons.png"))
    plt.close()

    # --------------------------------------------------
    # 2. FIRST CHIP + MASK OVERLAY
    # --------------------------------------------------
    chip = chips_dataset[0]
    chip_img = chip["image"].permute(1, 2, 0).numpy()
    chip_mask = chip["mask"][0].numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(chip_img)
    plt.imshow(chip_mask, alpha=0.5, cmap='Reds')
    plt.title("Chip with Mask Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "chip_mask_overlay.png"))
    plt.close()

    print(f"[Verification] Saved output images to: {save_dir}")




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

    eval = 0.1
    eval_size = int(len(chips_dataset) * eval)
    train_size = len(chips_dataset) - eval_size
    train_ds, eval_ds = torch.utils.data.random_split(chips_dataset, [train_size, eval_size])

    # 4 Channels: RGB + NIR
    UNet_Model = UNet(in_channels=4, out_channels=1 )

    print(f"Training on {len(chips_dataset)} chips...")

    save_verification_plots(sn7_dataset, chips_dataset)

    trained_model = train_model(
        dataset=train_ds,
        model=UNet_Model,
        epochs=50,
        batch_size=8,
        lr=1e-4,
        device=device,
        pos_weight=6,
        neg_weight=2,
        patience=10
    )

    #UNet_Model.load_state_dict(torch.load("unet_best.pth", map_location="cuda"))
    #UNet_Model.eval()

    evaluate_model(UNet_Model, eval_ds, device=device)