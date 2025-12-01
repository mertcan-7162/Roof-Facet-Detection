import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import ssl
import cv2
import numpy as np

# Custom modules
from dataset import MaskRefinementDataset
from model import get_shape_model
from utils import EarlyStopping, generate_validation_report

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


def load_config(config_path="polygon_estimator/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_transforms(height, width):
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
            ToTensorV2(),
        ],
    )


# --- LOSS FUNCTIONS ---
dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()


def criterion(prediction, target):
    """
    Combined loss:
    - 60% Dice Loss (shape consistency)
    - 40% BCE Loss (pixel accuracy)
    """
    d_loss = dice_loss_fn(prediction, target)
    b_loss = bce_loss_fn(prediction, target)
    return 0.6 * d_loss + 0.4 * b_loss


def train_one_epoch(loader, model, optimizer, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    epoch_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        predictions = model(data)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)


def validate(loader, model, device):
    model.eval()
    val_loss = 0
    dice_score = 0
    iou_score = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            predictions = model(data)

            loss = criterion(predictions, targets)
            val_loss += loss.item()

            preds_prob = torch.sigmoid(predictions)
            preds_bin = (preds_prob > 0.5).float()

            intersection = (preds_bin * targets).sum()
            dice_score += (2 * intersection) / ((preds_bin + targets).sum() + 1e-8)

            union = preds_bin.sum() + targets.sum() - intersection
            iou_score += (intersection + 1e-8) / (union + 1e-8)

    avg_loss = val_loss / len(loader)
    avg_dice = dice_score / len(loader)
    avg_iou = iou_score / len(loader)

    print(f"Validation -> Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")

    # Return Dice score for LR scheduler and Early Stopping
    return avg_dice


def main():
    # Load config
    cfg = load_config("polygon_estimator/config.yaml")

    if cfg['hyperparameters']['device'] == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = cfg['hyperparameters']['device']

    print(f"Device: {device}")

    # Paths
    base_dir = cfg['paths']['base_dir']

    train_inp_dir = os.path.join(base_dir, cfg['paths']['train_input_masks'])
    train_lbl_dir = os.path.join(base_dir, cfg['paths']['train_label_masks'])

    val_inp_dir = os.path.join(base_dir, cfg['paths']['valid_input_masks'])
    val_lbl_dir = os.path.join(base_dir, cfg['paths']['valid_label_masks'])

    # Dataset & DataLoader
    h, w = cfg['hyperparameters']['image_height'], cfg['hyperparameters']['image_width']

    train_ds = MaskRefinementDataset(train_inp_dir, train_lbl_dir, transform=get_transforms(h, w))
    val_ds = MaskRefinementDataset(val_inp_dir, val_lbl_dir, transform=get_transforms(h, w))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=cfg['hyperparameters']['num_workers']
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=cfg['hyperparameters']['num_workers']
    )

    print(f"Train Size: {len(train_ds)} | Val Size: {len(val_ds)}")

    # Model and optimizer
    model = get_shape_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['hyperparameters']['learning_rate'])

    # LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=cfg['control']['scheduler_factor'],
        patience=cfg['control']['scheduler_patience'],
        min_lr=cfg['control']['min_lr']
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=cfg['control']['early_stopping_patience'],
        verbose=True,
        path=cfg['paths']['checkpoint_save_path']
    )

    # Training Loop
    num_epochs = cfg['hyperparameters']['num_epochs']

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_one_epoch(train_loader, model, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_dice_score = validate(val_loader, model, device)

        scheduler.step(val_dice_score)
        early_stopping(val_dice_score, model)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print("Training Completed.")

    generate_validation_report(
        val_loader,
        model,
        device,
        num_samples=5,
        save_path="roof_estimator/final_validation_results.png"
    )


if __name__ == "__main__":
    main()