from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from UNet.UNet import UNet
from UNet.preprocessChips import BuildingChipsDataset

# ---------------- Metrics & Loss ---------------- #

def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2*intersection + smooth)/(union + smooth)
    return 1 - dice.mean()

def weighted_bce_dice_loss(preds, targets, pos_weight=5.0):
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(preds.device))(preds, targets)
    dice = dice_loss(preds, targets)
    return bce + dice

def pixel_accuracy(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    correct = (preds == targets).float()
    acc = correct.sum() / correct.numel()
    return acc.item()

def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

# ---------------- Training ---------------- #

def train_model(root, epochs=10, batch_size=4, lr=1e-4, device='cuda'):
    dataset = BuildingChipsDataset(root)
    val_size = int(0.1*len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        total_loss = 0
        total_acc = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = weighted_bce_dice_loss(preds, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += pixel_accuracy(preds, masks)

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)

        # ---------- Validation ----------
        model.eval()
        val_iou = 0
        val_acc = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_iou += iou_score(preds, masks)
                val_acc += pixel_accuracy(preds, masks)

        val_iou /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {avg_acc:.4f} | "
              f"Val IoU: {val_iou:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # ---------- Optional Visualization ----------
        if epoch % 5 == 0:
            imgs, masks = next(iter(val_loader))
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                preds = torch.sigmoid(model(imgs))
                pred_mask = (preds[0,0] > 0.5).cpu().numpy()
                plt.figure(figsize=(6,6))
                plt.imshow(imgs[0].cpu().permute(1,2,0))
                plt.imshow(pred_mask, alpha=0.5, cmap='Reds')
                plt.axis('off')
                plt.show()

    # Save model
    #torch.save(model.state_dict(), "unet_buildings.pth")
    #print("Training finished and model saved as unet_buildings.pth")
