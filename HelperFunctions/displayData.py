import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.patches import Polygon
import re

def load_image(path, bands=[1,2,3]):
    with rasterio.open(path) as src:
        img = src.read(bands).astype(np.float32)

        for i in range(img.shape[0]):
            band = img[i]
            band -= band.min()
            if band.max() > 0:
                band /= band.max()
            img[i] = band

        return img.transpose(1, 2, 0)

def display_images(sat_img, gdf):
    """
    Displays a satellite image and its building polygons side by side.

    Parameters:
    -----------
    sat_img : np.ndarray
        NumPy array of the satellite image (height, width, channels), scaled [0,1]
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with building polygons in pixel coordinates
    """
    height, width = sat_img.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # LEFT: satellite image
    axes[0].imshow(sat_img)
    axes[0].set_title("Satellite Image")
    axes[0].axis("on")

    # RIGHT: polygons only
    axes[1].set_xlim(0, sat_img.shape[1])
    axes[1].set_ylim(sat_img.shape[0], 0) 
    axes[1].set_aspect('equal')
    axes[1].set_title("Building Polygons")
    axes[1].axis("on")

    # Draw polygons in right panel
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            patch = Polygon(list(geom.exterior.coords), facecolor='none', edgecolor='fuchsia', linewidth=1)
            axes[1].add_patch(patch)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                patch = Polygon(list(poly.exterior.coords), facecolor='none', edgecolor='furchsia', linewidth=1)
                axes[1].add_patch(patch)

    fig.text(0.5, 0.01, f"Image dimensions: {width} x {height} (width x height)", 
             ha='center', fontsize=12, color='black')

    plt.show()

def get_date(file_path):
    """
    Extracts the date (YYYY-MM) from a SpaceNet image filename.

    Parameters:
    -----------
    file_path : str or Path
        Path to the image file (.tif)

    Returns:
    --------
    date_str : str
        Extracted date in format "YYYY-MM", or "Unknown" if not found.
    """

    file_path = Path(file_path)
    filename = file_path.stem 

    # Convert to YYYY_MM pattern
    match = re.search(r'(\d{4})_(\d{2})', filename)
    if match:
        year, month = match.groups()
        return f"{year}-{month}"
    else:
        return "Unknown"
