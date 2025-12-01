import numpy as np
from shapely.geometry import box
from shapely.affinity import translate
from rasterio.features import rasterize
import torch
from torch.utils.data import Dataset

class ChipsDataset(Dataset):
    def __init__(self, chips):
        self.chips = chips

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        img, mask, timestamp = self.chips[idx]

        return {
            "image": torch.tensor(img).permute(2,0,1).float(),
            "mask": torch.tensor(mask).unsqueeze(0).float(),
            "timestamp": timestamp
        }
    

def generate_chips_from_frame(frame, chip_size=256):
    """
    Input:
      frame = {
         "image": torch(C,H,W),
         "polygons_pix": [shapely polygon list]
      }
    Returns:
      list of (chip_img, chip_mask, timestamp)
    """

    img = frame["image"].permute(1,2,0).cpu().numpy()  # HWC
    H, W, _ = img.shape

    polygons = frame.get("polygons_pix", [])

    chips = []

    for y0 in range(0, H, chip_size):
        for x0 in range(0, W, chip_size):
            x1 = x0 + chip_size
            y1 = y0 + chip_size
            if x1 > W or y1 > H:
                continue

            chip_box = box(x0, y0, x1, y1)

            # polygons inside the chip
            polys_in = [g for g in polygons if g.intersects(chip_box)]

            # build mask
            if not polys_in:
                chip_mask = np.zeros((chip_size, chip_size), dtype=np.uint8)
            else:
                shifted = [
                    translate(g.intersection(chip_box), -x0, -y0)
                    for g in polys_in
                ]
                chip_mask = rasterize(
                    [(s, 1) for s in shifted],
                    out_shape=(chip_size, chip_size),
                    fill=0,
                    dtype=np.uint8
                )

            chip_img = img[y0:y1, x0:x1]
            chips.append((chip_img, chip_mask, frame["timestamp"]))

    return chips

def create_all_chips(sn7_dataset, chip_size=256):
    """
    sn7_dataset: instance of SpaceNet7Buildings
    returns: list of (chip_img, chip_mask)
    """

    all_chips = []
    for item in sn7_dataset:
        frames = item.get("frames", [item])
        for f in frames:
            chips = generate_chips_from_frame(f, chip_size)
            all_chips.extend(chips)
    return all_chips