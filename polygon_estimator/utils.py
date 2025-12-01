import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class EarlyStopping:
    """
    Stop training when validation score (Dice/IoU) does not improve.
    Works with 'Higher is Better' logic.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            verbose (bool): Whether to print logs.
            delta (float): Minimum change required to qualify as improvement.
            path (str): File path where the best model will be saved.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_score, model):
        """Evaluate new validation score and decide whether to stop or save."""
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(Best: {self.best_score:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        """Save model when validation score improves."""
        if self.verbose:
            print(
                f"Validation score improved ({self.val_score_max:.4f} → {val_score:.4f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_score_max = val_score


def generate_validation_report(loader, model, device, num_samples=5, save_path="final_validation_results.png"):
    """
    Generate a visual + metrics-based validation summary for segmentation models.
    
    Evaluates:
      - IoU (Edge)
      - IoU (Roof)
      - Dice (Edge)
      - Dice (Roof)

    Saves:
      - A grid visualization with Input | GT | Prediction for both classes.
    """
    print("\n" + "=" * 50)
    print("GENERATING FINAL VALIDATION REPORT")
    print("=" * 50)

    model.eval()

    total_edge_iou = 0
    total_roof_iou = 0
    total_edge_dice = 0
    total_roof_dice = 0
    num_batches = len(loader)

    saved_samples = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(loader, desc="Calculating Metrics")):
            images = images.to(device)

            # masks: [edge_mask, roof_mask]
            edge_target = masks[0].to(device)
            roof_target = masks[1].to(device)

            logits = model(images)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()

            # --- Metrics ---
            def batch_metrics(pred_tensor, target_tensor):
                pred_flat = pred_tensor.reshape(-1)
                target_flat = target_tensor.reshape(-1)

                intersection = (pred_flat * target_flat).sum()
                union = pred_flat.sum() + target_flat.sum() - intersection

                iou = (intersection + 1e-7) / (union + 1e-7)
                dice = (2 * intersection + 1e-7) / (pred_flat.sum() + target_flat.sum() + 1e-7)
                return iou.item(), dice.item()

            e_iou, e_dice = batch_metrics(preds[:, 0, :, :], edge_target)
            total_edge_iou += e_iou
            total_edge_dice += e_dice

            r_iou, r_dice = batch_metrics(preds[:, 1, :, :], roof_target)
            total_roof_iou += r_iou
            total_roof_dice += r_dice

            # --- Collect Samples For Visualization ---
            if len(saved_samples) < num_samples:
                batch_size = images.shape[0]
                for i in range(batch_size):
                    if len(saved_samples) >= num_samples:
                        break

                    img_cpu = images[i].cpu().permute(1, 2, 0).numpy()

                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_cpu = img_cpu * std + mean
                    img_cpu = np.clip(img_cpu, 0, 1)

                    saved_samples.append({
                        "img": img_cpu,
                        "gt_edge": edge_target[i].cpu().numpy(),
                        "pred_edge": preds[i, 0, :, :].cpu().numpy(),
                        "gt_roof": roof_target[i].cpu().numpy(),
                        "pred_roof": preds[i, 1, :, :].cpu().numpy()
                    })

    # --- Average Metrics ---
    avg_edge_iou = total_edge_iou / num_batches
    avg_edge_dice = total_edge_dice / num_batches
    avg_roof_iou = total_roof_iou / num_batches
    avg_roof_dice = total_roof_dice / num_batches

    # --- Print Summary Table ---
    print("\n" + "-" * 55)
    print(f"{'Metric Table':^55}")
    print("-" * 55)
    print(f"{'Class':<20} | {'IoU':<15} | {'Dice Score':<15}")
    print("-" * 55)
    print(f"{'Edge':<20} | {avg_edge_iou:.4f}          | {avg_edge_dice:.4f}")
    print(f"{'Roof (Area)':<20} | {avg_roof_iou:.4f}          | {avg_roof_dice:.4f}")
    print("-" * 55)
    print(f"Combined Dice Score: {(avg_edge_dice + avg_roof_dice) / 2:.4f}")
    print("-" * 55 + "\n")

    # --- Visualization ---
    print(f"Creating visualization grid with {len(saved_samples)} samples...")

    fig, axs = plt.subplots(
        nrows=len(saved_samples),
        ncols=5,
        figsize=(20, 4 * len(saved_samples))
    )

    if len(saved_samples) == 1:
        axs = axs[np.newaxis, :]

    for i, sample in enumerate(saved_samples):
        axs[i, 0].imshow(sample["img"])
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(sample["gt_roof"], cmap="gray")
        axs[i, 1].set_title("GT Roof Mask")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(sample["pred_roof"], cmap="gray")
        axs[i, 2].set_title("Pred Roof Mask")
        axs[i, 2].axis("off")

        axs[i, 3].imshow(sample["gt_edge"], cmap="gray")
        axs[i, 3].set_title("GT Edge Mask")
        axs[i, 3].axis("off")

        axs[i, 4].imshow(sample["pred_edge"], cmap="gray")
        axs[i, 4].set_title("Pred Edge Mask")
        axs[i, 4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Report saved successfully to: {save_path}")