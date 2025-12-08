from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ---------------- Metrics  ---------------- #

def compute_iou(preds, masks, threshold=0.5, eps=1e-6):
    preds_bin = (preds > threshold).float()
    intersection = (preds_bin * masks).sum(dim=[1,2,3])
    union = preds_bin.sum(dim=[1,2,3]) + masks.sum(dim=[1,2,3]) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def compute_accuracy(preds, masks, threshold=0.5):
    preds_bin = (preds > threshold).float()
    correct = (preds_bin == masks).float().mean()
    return correct.item()

def compute_pred_only_accuracy(preds, masks, threshold=0.5, eps=1e-6):
    """
    Computes accuracy only over pixels where model predicts 1.
    Ignores pixels predicted as 0.
    """
    preds_bin = (preds > threshold).float()
    # Only consider pixels predicted as 1
    mask_pred_1 = preds_bin == 1
    if mask_pred_1.sum() == 0:
        return 0.0  # avoid division by zero if no positives predicted
    correct = (preds_bin[mask_pred_1] == masks[mask_pred_1]).float().mean()
    return correct.item()

def compute_dice(preds, masks, threshold=0.5, eps=1e-6):
    preds_bin = (preds > threshold).float()
    intersection = (preds_bin * masks).sum(dim=[1,2,3])
    dice = (2 * intersection + eps) / (
        preds_bin.sum(dim=[1,2,3]) + masks.sum(dim=[1,2,3]) + eps
    )
    return dice.mean().item()

# --------- Combined Weighted Loss --------- #
def weighted_bce_dice_loss(preds, masks, pos_weight=5, neg_weight=1, eps=1e-6):
    """
    preds: logits (B,1,H,W)
    masks: ground truth masks (B,1,H,W)
    pos_weight: weight for false negatives
    neg_weight: weight for false positives
    """
    probs = torch.sigmoid(preds)

    # Weighted BCE
    bce_loss = -(
        pos_weight * masks * torch.log(probs + 1e-8) +
        neg_weight * (1 - masks) * torch.log(1 - probs + 1e-8)
    ).mean()

    # Dice loss
    intersection = (probs * masks).sum(dim=[1,2,3])
    dice_score = (2 * intersection + eps) / (probs.sum(dim=[1,2,3]) + masks.sum(dim=[1,2,3]) + eps)
    dice_loss = 1 - dice_score.mean()

    return bce_loss + dice_loss

# -------------- Visualization ------------- #

def visualize_predictions(img_np, mask_true, mask_pred):
    """
    img_np: HWC numpy image (RGB)
    mask_true: ground truth mask (H,W) binary
    mask_pred: predicted mask (H,W) binary
    """
    # Compute error map
    if img_np.shape[2] > 3:
        img_np = img_np[:, :, :3]

    tp = (mask_true == 1) & (mask_pred == 1)  # correctly predicted
    fn = (mask_true == 1) & (mask_pred == 0)  # missed
    fp = (mask_true == 0) & (mask_pred == 1)  # extra

    # Create RGB overlay
    overlay = np.zeros((*mask_true.shape, 3), dtype=np.float32)

    overlay[..., 0] = fn  # Red = missed pixels
    overlay[..., 1] = tp  # Green = correct pixels
    overlay[..., 2] = fp  # Blue = extra pixels

    # Blend with original image
    img_float = img_np.astype(np.float32) / 255.0 if img_np.max() > 1 else img_np
    blended = 0.5 * img_float + 0.5 * overlay

    plt.figure(figsize=(6,6))
    plt.imshow(blended)
    plt.title("Prediction Errors (Red=FN, Green=TP, Blue=FP)")
    plt.axis('off')

    plt.show(block=False)
    plt.pause(30)
    plt.close()

# ---------------- Training ---------------- #

def train_model(dataset, model, epochs=20, batch_size=4, lr=1e-4,
                device='cuda', pos_weight=5, neg_weight=1, patience=5):
    
    pos_weight = torch.tensor(pos_weight).to(device)
    neg_weight = torch.tensor(neg_weight).to(device)

    # Load dataset for training
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    model.to(device)

    best_val_iou = 0
    epochs_no_improve = 0

    # --- Track accuracy at each batch ---
    acc_history = []
    loss_history = []
    batch_indices = []
    batch_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_iou  = 0
        epoch_acc  = 0
        epoch_pred_acc = 0
        epoch_dice = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=90)
        for batch in pbar:

            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = weighted_bce_dice_loss(preds, masks, pos_weight=pos_weight, neg_weight=neg_weight)
            loss.backward()
            optimizer.step()

            preds_sig = torch.sigmoid(preds)

            iou  = compute_iou(preds_sig, masks)
            acc  = compute_accuracy(preds_sig, masks)
            pred_acc = compute_pred_only_accuracy(preds_sig, masks)
            dice = compute_dice(preds_sig, masks)

            epoch_loss += loss.item()
            epoch_iou  += iou
            epoch_acc  += acc
            epoch_pred_acc += pred_acc
            epoch_dice += dice

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "IoU": f"{iou:.3f}",
                "Acc": f"{acc:.3f}",
                "PredAcc": f"{pred_acc:.3f}",
                "Dice": f"{dice:.3f}",
            })

            # ------ NEW: record accuracy per batch ------
            acc_history.append(acc)
            loss_history.append(loss.item())
            batch_indices.append(batch_counter)
            batch_counter += 1

        n = len(train_loader)
        print(f"\nEpoch {epoch+1}: "
              f"Loss={epoch_loss/n:.4f}, "
              f"IoU={epoch_iou/n:.3f}, "
              f"Acc={epoch_acc/n:.3f}, "
              f"PredAcc={epoch_pred_acc/n:.3f}, "
              f"Dice={epoch_dice/n:.3f}")

        # ---------- Validation & Visualization ----------
        model.eval()
        val_iou, val_acc, val_dice, val_pred_acc = 0, 0, 0, 0
        n_val_batches = 0

        with torch.no_grad():
            for val_batch in val_loader:
                val_imgs = val_batch["image"].to(device)
                val_masks = val_batch["mask"].to(device)
                val_preds = torch.sigmoid(model(val_imgs))

                val_iou  += compute_iou(val_preds, val_masks)
                val_acc  += compute_accuracy(val_preds, val_masks)
                val_pred_acc += compute_pred_only_accuracy(val_preds, val_masks)
                val_dice += compute_dice(val_preds, val_masks)
                n_val_batches += 1

        val_iou  /= n_val_batches
        val_acc  /= n_val_batches
        val_pred_acc /= n_val_batches
        val_dice /= n_val_batches

        print(f"Validation metrics after Epoch {epoch+1}: "
            f"IoU={val_iou:.3f}, "
            f"Accuracy={val_acc:.3f}, "
            f"PredAcc={val_pred_acc:.3f}, "
            f"Dice={val_dice:.3f}")
        
        scheduler.step(val_iou)

        # ---------- Early Stopping ----------
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), "unet_best.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training complete.")

    # ----------------------------
    # Plot Accuracy and Loss Over Time
    # ----------------------------

    # Save plots
    os.makedirs("plots", exist_ok=True)

    # Accuracy plot
    plt.figure(figsize=(12,4))
    plt.plot(batch_indices, acc_history, linewidth=1)
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Time")
    plt.savefig("plots/accuracy_over_time.png")
    plt.close()

    # Loss plot
    plt.figure(figsize=(12,4))
    plt.plot(batch_indices, loss_history, linewidth=1)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.savefig("plots/loss_over_time.png")
    plt.close()

    return model
