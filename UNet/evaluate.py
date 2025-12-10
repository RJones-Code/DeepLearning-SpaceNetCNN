import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


plt.switch_backend("Agg")

def evaluate_model(model, dataset, device="cpu", save_dir="evaluation_outputs"):
    """
    Runs evaluation on a dataset of chips.
    Produces:
      - histogram of predicted probabilities
      - confusion matrix heatmap
      - example predictions saved to disk
    """
    os.makedirs(save_dir, exist_ok=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    all_probs = []
    all_preds = []
    all_labels = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(imgs)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.append(probs.cpu().numpy().reshape(-1))
            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_labels.append(masks.cpu().numpy().reshape(-1))

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print(f"[Eval] Collected {len(all_preds)} pixel predictions.")

    # 1. HISTOGRAM OF PREDICTED PROBABILITIES
    plt.figure(figsize=(8,5))
    plt.hist(all_probs, bins=30, alpha=0.75)
    plt.title("Histogram of Predictions (Flattened)")
    plt.xlabel("Predicted Value (0/1)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "histogram_predictions.png"))
    plt.close()

    # 2. CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    print("[Eval] Saved confusion matrix & histogram.")

    # 3. SAVE EXAMPLE PREDICTIONS
    print("[Eval] Saving example predictions...")

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(loader):
        if i >= 6:
            break  # Save only a few examples

        img = batch["image"].to(device)
        mask = batch["mask"].to(device)

        prob = torch.sigmoid(model(img))[0,0].detach().cpu().numpy()
        pred = (prob > 0.5).astype(np.float32)

        mask_np = mask[0,0].cpu().numpy()

        img_np = img[0].cpu().permute(1,2,0).numpy()

        if img_np.shape[-1] > 3:
            img_np = img_np[:, :, :3]

        tp = (mask_np == 1) & (pred == 1)
        fn = (mask_np == 1) & (pred == 0)
        fp = (mask_np == 0) & (pred == 1)

        overlay = np.zeros((*mask_np.shape, 3), dtype=np.float32)
        overlay[..., 0] = fn 
        overlay[..., 1] = tp  
        overlay[..., 2] = fp 

        img_float = img_np.astype(np.float32) / 255.0 if img_np.max() > 1 else img_np
        blended = 0.5 * img_float + 0.5 * overlay

        plt.figure(figsize=(6,6))
        plt.imshow(blended)
        plt.imshow(pred, alpha=0.4, cmap="Reds")
        plt.title("Prediction Overlay (Red=FN, Green=TP, Blue=FP)")
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"prediction_overlay_{i}.png"))
        plt.close()

    print("[Eval] Example predictions saved.")
