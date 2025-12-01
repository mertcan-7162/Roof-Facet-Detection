import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_score, model):
        
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best: {self.best_score:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            self.trace_func(f'Validation score improved ({self.best_score:.4f} --> {score:.4f}).  Saving model ')
            self.best_score = score
        torch.save(model.state_dict(), self.path)


def validate(loader, model, device):
    model.eval()

    dice_edge_sum = 0
    dice_area_sum = 0

    iou_edge_sum = 0
    iou_area_sum = 0

    with torch.no_grad():
        for image, masks in loader:
            image = image.to(device)

            edge_mask = masks[0].to(device).float()
            roof_mask = masks[1].to(device).float()

            # --- Forward ---
            logits = model(image)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()

            # ==========================
            #    EDGE METRICS
            # ==========================
            edge_preds = preds[:, 0, :, :]

            # DICE
            intersection_edge = (edge_preds * edge_mask).sum()
            union_edge = (edge_preds + edge_mask).sum()
            dice_edge = (2 * intersection_edge) / (union_edge + 1e-8)
            dice_edge_sum += dice_edge

            # IoU
            # IoU = intersection / (A + B - intersection)
            iou_edge = intersection_edge / (union_edge - intersection_edge + 1e-8)
            iou_edge_sum += iou_edge

            # ==========================
            #    AREA METRICS
            # ==========================
            roof_preds = preds[:, 1, :, :]

            # DICE
            intersection_area = (roof_preds * roof_mask).sum()
            union_area = (roof_preds + roof_mask).sum()
            dice_area = (2 * intersection_area) / (union_area + 1e-8)
            dice_area_sum += dice_area

            # IoU
            iou_area = intersection_area / (union_area - intersection_area + 1e-8)
            iou_area_sum += iou_area

    # Averaging
    avg_dice_edge = dice_edge_sum / len(loader)
    avg_dice_area = dice_area_sum / len(loader)

    avg_iou_edge = iou_edge_sum / len(loader)
    avg_iou_area = iou_area_sum / len(loader)

    print(
        f"Validation -> "
        f"Edge Dice: {avg_dice_edge:.4f}, Area Dice: {avg_dice_area:.4f} | "
        f"Edge IoU: {avg_iou_edge:.4f}, Area IoU: {avg_iou_area:.4f}"
    )

    # Combined score 
    combined = (
        avg_dice_edge + avg_dice_area +
        avg_iou_edge + avg_iou_area
    ) / 4

    return combined



def generate_validation_report(loader, model, device, num_samples=5, save_path="final_validation_results.png"):
    print("\n" + "="*50)
    print("GENERATING FINAL VALIDATION REPORT")
    print("="*50)
    
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
            edge_target = masks[0].to(device)
            roof_target = masks[1].to(device)

            logits = model(images)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()
            
            # --- Batch Metrics Calculation ---
            # Helper function for IoU and Dice (Batch level)
            def calculate_batch_metrics(pred_tensor, target_tensor):
                # Flatten tensors for calculation
                pred_flat = pred_tensor.reshape(-1)
                target_flat = target_tensor.reshape(-1)
                
                intersection = (pred_flat * target_flat).sum()
                union = pred_flat.sum() + target_flat.sum() - intersection
                
                iou = (intersection + 1e-7) / (union + 1e-7)
                dice = (2 * intersection + 1e-7) / (pred_flat.sum() + target_flat.sum() + 1e-7)
                return iou.item(), dice.item()

            # Edge Metrics (Channel 0)
            e_iou, e_dice = calculate_batch_metrics(preds[:, 0, :, :], edge_target)
            total_edge_iou += e_iou
            total_edge_dice += e_dice

            # Roof Metrics (Channel 1)
            r_iou, r_dice = calculate_batch_metrics(preds[:, 1, :, :], roof_target)
            total_roof_iou += r_iou
            total_roof_dice += r_dice
            
            # --- Sample Collection for Visualization ---
            if len(saved_samples) < num_samples:
                batch_size = images.shape[0]
                for i in range(batch_size):
                    if len(saved_samples) >= num_samples:
                        break
                    
                    # Tensor -> Numpy Conversion (Denormalization for visualization)
                    img_cpu = images[i].cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_cpu = img_cpu * std + mean
                    img_cpu = np.clip(img_cpu, 0, 1)
                    
                    # Masks (GT & Pred)
                    gt_edge = edge_target[i].cpu().numpy()
                    pred_edge = preds[i, 0, :, :].cpu().numpy()
                    
                    gt_roof = roof_target[i].cpu().numpy()
                    pred_roof = preds[i, 1, :, :].cpu().numpy()
                    
                    saved_samples.append({
                        'img': img_cpu,
                        'gt_edge': gt_edge,
                        'pred_edge': pred_edge,
                        'gt_roof': gt_roof,
                        'pred_roof': pred_roof
                    })

    avg_edge_iou = total_edge_iou / num_batches
    avg_edge_dice = total_edge_dice / num_batches
    avg_roof_iou = total_roof_iou / num_batches
    avg_roof_dice = total_roof_dice / num_batches

    print("\n" + "-"*55)
    print(f"{'Metric Table':^55}")
    print("-" * 55)
    print(f"{'Class':<20} | {'IoU':<15} | {'Dice Score':<15}")
    print("-" * 55)
    print(f"{'Edge':<20} | {avg_edge_iou:.4f}          | {avg_edge_dice:.4f}")
    print(f"{'Roof (Area)':<20} | {avg_roof_iou:.4f}          | {avg_roof_dice:.4f}")
    print("-" * 55)
    print(f"Combined Dice Score: {(avg_edge_dice + avg_roof_dice)/2:.4f}")
    print("-" * 55 + "\n")


    print(f"Creating visualization grid with {len(saved_samples)} samples...")
    
    fig, axs = plt.subplots(nrows=len(saved_samples), ncols=5, figsize=(20, 4 * len(saved_samples)))
    
    if len(saved_samples) == 1:
        axs = axs[np.newaxis, :]

    for i, sample in enumerate(saved_samples):
        # Column 1: Input Image
        axs[i, 0].imshow(sample['img'])
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")
        
        # Column 2: GT Roof
        axs[i, 1].imshow(sample['gt_roof'], cmap="gray")
        axs[i, 1].set_title("GT Roof Mask")
        axs[i, 1].axis("off")
        
        # Column 3: Pred Roof
        axs[i, 2].imshow(sample['pred_roof'], cmap="gray")
        axs[i, 2].set_title("Pred Roof Mask")
        axs[i, 2].axis("off")

        # Column 4: GT Edge
        axs[i, 3].imshow(sample['gt_edge'], cmap="gray")
        axs[i, 3].set_title("GT Edge Mask")
        axs[i, 3].axis("off")

        # Column 5: Pred Edge
        axs[i, 4].imshow(sample['pred_edge'], cmap="gray")
        axs[i, 4].set_title("Pred Edge Mask")
        axs[i, 4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Report saved successfully to: {save_path}")
