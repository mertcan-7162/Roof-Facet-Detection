import os
import cv2
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ssl
from tqdm import tqdm

from model import get_model

ssl._create_default_https_context = ssl._create_unverified_context


def load_config(config_path="polygon_estimator/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_transforms(height, width):
    """Preprocessing pipeline for single-channel mask input."""
    return A.Compose([
        A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        ToTensorV2(),
    ])


def calculate_metrics(pred, target):
    """Compute IoU and Dice score for a binary mask."""
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)

    return iou, dice


def main():

    cfg = load_config("polygon_estimator/config.yaml")

    if cfg['hyperparameters']['device'] == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available()
            else "cpu")
        )
    else:
        device = cfg['hyperparameters']['device']

    print(f"Device: {device}")

    model = get_model(cfg).to(device)
    checkpoint_path = cfg['paths']['checkpoint_save_path']

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint at {checkpoint_path}\n{e}")
        return

    model.eval()

    base_dir = cfg['paths']['base_dir']

    test_input_mask_dir = os.path.join(base_dir, cfg['paths']['test_input_masks'])
    test_label_mask_dir = os.path.join(base_dir, cfg['paths']['test_label_masks'])

    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}
    all_masks = sorted([
        f for f in os.listdir(test_input_mask_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    num_samples = min(3, len(all_masks))
    selected_masks = random.sample(all_masks, num_samples)

    h, w = cfg['hyperparameters']['image_height'], cfg['hyperparameters']['image_width']
    transform = get_transforms(h, w)

    metrics = {"edge_iou": [], "edge_dice": []}
    visuals = []

    print(f"\nProcessing {num_samples} mask samples...")

    with torch.no_grad():
        for mask_name in selected_masks:

            input_mask_path = os.path.join(test_input_mask_dir, mask_name)
            input_mask_img = cv2.imread(input_mask_path, 0)
            if input_mask_img is None:
                continue

            orig_h, orig_w = input_mask_img.shape[:2]

            gt_mask_path = os.path.join(test_label_mask_dir, mask_name)
            gt_edge = cv2.imread(gt_mask_path, 0)
            if gt_edge is None:
                gt_edge = np.zeros((orig_h, orig_w), dtype=np.uint8)

            input_mask_float = input_mask_img.astype(np.float32) / 255.0
            gt_edge_bin = (gt_edge > 127).astype(np.float32)

            aug = transform(image=input_mask_float)
            tensor_img = aug["image"].unsqueeze(0).to(device)

            logits = model(tensor_img)
            probs = torch.sigmoid(logits)

            pred_edge_prob = probs[0, 0, :, :].cpu().numpy()
            pred_edge_bin = (pred_edge_prob > 0.5).astype(np.float32)

            e_iou, e_dice = calculate_metrics(pred_edge_bin, gt_edge_bin)

            metrics["edge_iou"].append(e_iou)
            metrics["edge_dice"].append(e_dice)

            visuals.append({
                "input_mask": input_mask_float,
                "gt_edge": gt_edge_bin,
                "pred_edge": pred_edge_bin,
            })

    print("\n=============================================")
    print(f" SHAPE REFINEMENT EVALUATION ({len(metrics['edge_dice'])} samples)")
    print("=============================================")
    print(f"Dice Score : {np.mean(metrics['edge_dice']):.4f}")
    print(f"IoU Score  : {np.mean(metrics['edge_iou']):.4f}")
    print("=============================================")

    if visuals:
        print("Generating visualization...")

        rows = len(visuals)
        cols = 3

        fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        if rows == 1:
            axs = axs[np.newaxis, :]

        for i, viz in enumerate(visuals):

            axs[i, 0].imshow(viz['input_mask'].squeeze(), cmap='gray')
            axs[i, 0].set_title("Input Mask")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(viz['gt_edge'].squeeze(), cmap='gray')
            axs[i, 1].set_title("Ground Truth")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(viz['pred_edge'].squeeze(), cmap='gray')
            axs[i, 2].set_title("Predicted Mask")
            axs[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig("polygon_estimator/refinement_quick_check.png", dpi=150)
        print("Visualization saved to polygon_estimator/refinement_quick_check.png")

    print("\nDone.")


if __name__ == "__main__":
    main()