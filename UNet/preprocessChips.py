import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import translate
from rasterio.features import rasterize

class BuildingChipsDataset(Dataset):
    def __init__(self, root, chip_size=256, min_area=10, bands=[1,2,3]):
        self.root = Path(root)
        self.chip_size = chip_size
        self.min_area = min_area
        self.bands = bands
        self.samples = []

        # Iterate over tiles
        for tile in self.root.iterdir():
            if not tile.is_dir():
                continue
            images_dir = tile / "images_masked"
            labels_dir = tile / "labels_match_pix"
            for img_path in images_dir.glob("*.tif"):
                label_path = labels_dir / f"{img_path.stem}_Buildings.geojson"
                if label_path.exists():
                    self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image
        img = np.array(Image.open(img_path).convert("RGB")) / 255.0  # normalize

        # Load building polygons
        gdf = gpd.read_file(label_path)

        gdf = gdf.set_crs(None, allow_override=True)
        gdf = gdf[gdf.area >= self.min_area]

        chips = self.generate_chips(img, gdf)

        # pick a random chip with at least one building
        chips_with_buildings = [c for c in chips if c[1].sum() > 0]
        if not chips_with_buildings:
            # fallback if no building in any chip
            chip_img, chip_mask, _ = chips[np.random.randint(len(chips))]
        else:
            chip_img, chip_mask, _ = chips_with_buildings[np.random.randint(len(chips_with_buildings))]

        # Convert to tensor
        chip_img = torch.from_numpy(chip_img.transpose(2,0,1)).float()  # (C,H,W)
        chip_mask = torch.from_numpy(chip_mask).unsqueeze(0).float()      # (1,H,W)

        return chip_img, chip_mask

    def generate_chips(self, img, gdf):
        H, W, _ = img.shape
        chips = []

        for y0 in range(0, H, self.chip_size):
            for x0 in range(0, W, self.chip_size):
                x1 = min(x0 + self.chip_size, W)
                y1 = min(y0 + self.chip_size, H)

                chip_img = img[y0:y1, x0:x1]
                if chip_img.shape[0] != self.chip_size or chip_img.shape[1] != self.chip_size:
                    continue

                chip_box = box(x0, y0, x1, y1)
                gdf_chip = gdf[gdf.intersects(chip_box)].copy()

                if gdf_chip.empty:
                    chip_mask = np.zeros((self.chip_size, self.chip_size), dtype=np.uint8)
                    gdf_shifted = gpd.GeoDataFrame(geometry=[])
                else:
                    shifted_geoms = [translate(geom.intersection(chip_box), xoff=-x0, yoff=-y0) for geom in gdf_chip.geometry]
                    gdf_shifted = gpd.GeoDataFrame(geometry=shifted_geoms, crs=None)

                    chip_mask = rasterize(
                        [(geom, 1) for geom in shifted_geoms],
                        out_shape=(self.chip_size, self.chip_size),
                        fill=0,
                        dtype=np.uint8
                    )

                chips.append((chip_img, chip_mask, gdf_shifted))
        return chips
