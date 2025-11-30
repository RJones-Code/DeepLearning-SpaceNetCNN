import os
import re
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
from torchvision import transforms


# Regex to extract timestamp from filenames
TIMESTAMP_RE = re.compile(r"global_monthly_(\d{4}_\d{2})_mosaic")


class SpaceNet7Buildings(Dataset):
    def __init__(self, root, transform=None, return_series=False):
        """
        root: 'workspace/archive/SN7_buildings_train/train'
        return_series: 
            False → returns one random timestamp per tile
            True  → returns all timestamps for that tile
        """
        self.root = root
        self.transform = transform
        self.return_series = return_series
        
        # Each subfolder is a tile (e.g., L15-....)
        self.tiles = sorted([
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        # Pre-index timestamps per tile
        self.index = []
        for tile_path in self.tiles:
            timestamps = self._collect_timestamps(tile_path)

            if return_series:
                # One dataset item = one tile (all timestamps)
                self.index.append((tile_path, timestamps))
            else:
                # One item per timestamp
                for ts in timestamps:
                    self.index.append((tile_path, [ts]))

    def _collect_timestamps(self, tile_path):
        """Return sorted timestamps like ['2018_01', '2018_02', ...]"""
        img_dir = os.path.join(tile_path, "images")
        timestamps = []

        for fname in os.listdir(img_dir):
            match = TIMESTAMP_RE.search(fname)
            if match:
                timestamps.append(match.group(1))

        return sorted(list(set(timestamps)))

    def __len__(self):
        return len(self.index)

    def _load_image(self, path):
        # SpaceNet-7 images are georeferenced TIFFs → rasterio handles them best
        with rasterio.open(path) as src:
            img = src.read()       # shape: (C, H, W)
            img = torch.from_numpy(img).float() / 255.0
        return img

    def _load_geojson(self, path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None

    def _load_frame(self, tile_path, timestamp):
        t = timestamp  # e.g., '2018_03'
        
        # Build wildcard match prefix
        prefix = f"global_monthly_{t}_mosaic"

        # Paths
        img_dir   = os.path.join(tile_path, "images")
        mask_dir  = os.path.join(tile_path, "images_masked")
        lbl_dir   = os.path.join(tile_path, "labels")
        lblm_dir  = os.path.join(tile_path, "labels_match")
        pix_dir   = os.path.join(tile_path, "labels_match_pix")

        # Find matching files
        def find_file(folder, pattern):
            for f in os.listdir(folder):
                if pattern in f:
                    return os.path.join(folder, f)
            return None

        img_path   = find_file(img_dir,  prefix)
        mask_path  = find_file(mask_dir, prefix)
        lbl_path   = find_file(lbl_dir,  prefix)
        lblm_path  = find_file(lblm_dir, prefix)
        pix_path   = find_file(pix_dir,  prefix)

        # Load images
        image = self._load_image(img_path) if img_path else None
        masked = self._load_image(mask_path) if mask_path else None

        if self.transform and image is not None:
            image = self.transform(image)
        if self.transform and masked is not None:
            masked = self.transform(masked)

        # Load annotations
        labels = self._load_geojson(lbl_path)
        labels_match = self._load_geojson(lblm_path)
        labels_match_pix = self._load_geojson(pix_path)

        return {
            "timestamp": timestamp,
            "image": image,
            "image_masked": masked,
            "labels": labels,
            "labels_match": labels_match,
            "labels_match_pix": labels_match_pix
        }

    def __getitem__(self, idx):
        tile_path, timestamps = self.index[idx]

        frames = [self._load_frame(tile_path, t) for t in timestamps]

        if self.return_series:
            return {
                "tile": os.path.basename(tile_path),
                "frames": frames
            }
        else:
            return frames[0]