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
import numpy as np
import matplotlib as plt 

from dataset import RoofDataset
from model import get_model
from utils import EarlyStopping, generate_validation_report, validate

ssl._create_default_https_context = ssl._create_unverified_context

def load_config(config_path="roof_estimator/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_transforms(height, width):
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=height, 
                min_width=width, 
                border_mode=0,
                value=0, 
                mask_value=0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

# --- LOSS FUNCTION ---

dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
focal_loss_fn = smp.losses.FocalLoss(mode="binary")
bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()

def criterion(predictions, targets, device):

    edge_pred = predictions[:, 0, :, :]    
    roof_pred = predictions[:, 1, :, :]

    edge_target = targets[0].to(device)
    roof_target = targets[1].to(device)

    edge_dice = dice_loss_fn(edge_pred, edge_target)
    
    if device == "mps":
         edge_pred = edge_pred.to("cpu") 
         edge_target = edge_target.to("cpu") 
    edge_focal = focal_loss_fn(edge_pred.reshape(-1), edge_target.reshape(-1))
    
    edge_loss = 0.7 * edge_dice + 0.3 * edge_focal

    roof_dice = dice_loss_fn(roof_pred, roof_target) 
    roof_bce = bce_loss_fn(roof_pred, roof_target)

    roof_loss = 0.8 * roof_dice + 0.2 * roof_bce 

    return 0.6 * edge_loss + 0.4 * roof_loss

def train_one_epoch(loader, model, optimizer, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    epoch_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        
        # Forward
        predictions = model(data) 
        
        # Loss
        loss = criterion(predictions, targets, device)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)

def main():

    cfg = load_config("roof_estimator/config.yaml")
    
    if cfg['hyperparameters']['device'] == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    else:
        device = cfg['hyperparameters']['device']
        
    print(f"Using Device: {device}")

    base_dir = cfg['paths']['base_dir']
    train_img_dir = os.path.join(base_dir, cfg['paths']['train_images'])
    train_edge_dir = os.path.join(base_dir, cfg['paths']['train_edge_masks'])
    train_roof_dir = os.path.join(base_dir, cfg['paths']['train_roof_masks'])
    
    valid_img_dir = os.path.join(base_dir, cfg['paths']['valid_images'])
    valid_edge_dir = os.path.join(base_dir, cfg['paths']['valid_edge_masks'])
    valid_roof_dir = os.path.join(base_dir, cfg['paths']['valid_roof_masks'])

    # Dataset & Dataloader
    h, w = cfg['hyperparameters']['image_height'], cfg['hyperparameters']['image_width']
    batch_size = cfg['hyperparameters']['batch_size']
    num_workers = cfg['hyperparameters']['num_workers']

    train_ds = RoofDataset(train_img_dir, train_edge_dir, train_roof_dir, transform=get_transforms(h, w))
    val_ds = RoofDataset(valid_img_dir, valid_edge_dir, valid_roof_dir, transform=get_transforms(h, w))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train Size: {len(train_ds)} | Val Size: {len(val_ds)}")

    # Model & Optimizer
    model = get_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['hyperparameters']['learning_rate'])

    # Scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=cfg['training_control']['scheduler_factor'], 
        patience=cfg['training_control']['scheduler_patience'], 
        min_lr=cfg['training_control']['min_lr']
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=cfg['training_control']['early_stopping_patience'],
        verbose=True,
        path=cfg['paths']['checkpoint_save_path']
    )

    # Training Loop
    num_epochs = cfg['hyperparameters']['num_epochs']

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_loss = train_one_epoch(train_loader, model, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_score = validate(val_loader, model, device)
        print(f"Combined Val Score: {val_score:.4f}")

        scheduler.step(val_score)

        early_stopping(val_score, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
    generate_validation_report(
        val_loader, 
        model, 
        device, 
        num_samples=5, 
        save_path="roof_estimator/final_validation_results.png"
    )
    
if __name__ == "__main__":
    main()