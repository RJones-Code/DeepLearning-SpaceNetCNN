import os
from dotenv import load_dotenv
import torch

from UNet.train import train_model as train

if __name__ == "__main__":
    load_dotenv()  # load variables from .env

    root = os.getenv("SPACENET_ROOT")
    if root is None:
        raise ValueError("SPACENET_ROOT not set in .env!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(
        root=root,
        epochs=20,
        batch_size=4,
        lr=1e-4,
        device=device,
    )