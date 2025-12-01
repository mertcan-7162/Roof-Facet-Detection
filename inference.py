import os
import cv2
import yaml
import torch
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(encoder, in_channels, classes, checkpoint_path, device):
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None, 
        in_channels=in_channels,
        classes=classes, 
        activation=None
    )
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit(1)
        
    model.to(device)
    model.eval()
    return model

def get_preprocessing(height, width):
    return A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])

def calculate_metrics(pred_mask, gt_mask):

    pred = (pred_mask > 127).astype(np.float32)
    gt = (gt_mask > 127).astype(np.float32)
    
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (pred.sum() + gt.sum() + 1e-7)
    
    return iou, dice

def random_colors(N):
    colors = []
    for _ in range(N):
        c = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        colors.append(c)
    return colors

def add_title(img, text):
    h, w = img.shape[:2]
    canvas = np.zeros((h + 40, w, 3), dtype=np.uint8)
    canvas[40:, :] = img
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas

def create_metrics_summary(metrics, output_path):

    avg_roof_iou = np.mean(metrics['roof_iou']) if metrics['roof_iou'] else 0
    avg_roof_dice = np.mean(metrics['roof_dice']) if metrics['roof_dice'] else 0
    
    avg_raw_edge_iou = np.mean(metrics['raw_edge_iou']) if metrics['raw_edge_iou'] else 0
    avg_raw_edge_dice = np.mean(metrics['raw_edge_dice']) if metrics['raw_edge_dice'] else 0
    
    avg_edge_iou = np.mean(metrics['edge_iou']) if metrics['edge_iou'] else 0
    avg_edge_dice = np.mean(metrics['edge_dice']) if metrics['edge_dice'] else 0
    
    height, width = 550, 600
    summary_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (50, 50, 50) 
    
    cv2.putText(summary_img, "Inference Metrics Summary", (50, 50), font, 1, (0, 0, 150), 2)
    cv2.line(summary_img, (50, 65), (550, 65), (0, 0, 0), 2)
    
    cv2.putText(summary_img, f"Roof Segmentation:", (50, 120), font, 0.8, (0, 0, 0), 2)
    cv2.putText(summary_img, f"  IoU: {avg_roof_iou:.4f}", (50, 160), font, 0.7, color, 2)
    cv2.putText(summary_img, f"  Dice: {avg_roof_dice:.4f}", (50, 200), font, 0.7, color, 2)
    
    cv2.putText(summary_img, f"Raw Edge (Model 1):", (50, 260), font, 0.8, (0, 0, 0), 2)
    cv2.putText(summary_img, f"  IoU: {avg_raw_edge_iou:.4f}", (50, 300), font, 0.7, color, 2)
    cv2.putText(summary_img, f"  Dice: {avg_raw_edge_dice:.4f}", (50, 340), font, 0.7, color, 2)

    cv2.putText(summary_img, f"Refined Edge (Model 2):", (50, 400), font, 0.8, (0, 0, 0), 2)
    cv2.putText(summary_img, f"  IoU: {avg_edge_iou:.4f}", (50, 440), font, 0.7, color, 2)
    cv2.putText(summary_img, f"  Dice: {avg_edge_dice:.4f}", (50, 480), font, 0.7, color, 2)
    
    cv2.imwrite(output_path, summary_img)
    print(f"\n Metrics summary saved to: {output_path}")
    print(f"   Roof -> IoU: {avg_roof_iou:.4f}, Dice: {avg_roof_dice:.4f}")
    print(f"   Raw Edge -> IoU: {avg_raw_edge_iou:.4f}, Dice: {avg_raw_edge_dice:.4f}")
    print(f"   Refined Edge -> IoU: {avg_edge_iou:.4f}, Dice: {avg_edge_dice:.4f}")

def main():
    cfg = load_config()
    
    if cfg['device'] == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = cfg['device']

    # Load Models
    model1 = load_model(
        cfg['model1']['encoder'], 
        cfg['model1']['in_channels'], 
        cfg['model1']['classes'], 
        cfg['model1']['path'], 
        device
    )
    
    model2 = load_model(
        cfg['model2']['encoder'], 
        cfg['model2']['in_channels'], 
        cfg['model2']['classes'], 
        cfg['model2']['path'], 
        device
    )

    # Paths
    base_dir = cfg['paths']['base_dir']
    input_image_dir = os.path.join(base_dir, cfg['paths']['test_images'])
    gt_edge_dir = os.path.join(base_dir, cfg['paths']['test_edge_masks'])
    gt_roof_dir = os.path.join(base_dir, cfg['paths']['test_roof_masks'])
    
    output_dir = cfg['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = {'.png', '.jpg', '.jpeg', '.tif'}
    images = sorted([f for f in os.listdir(input_image_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    # Process only 5 random images for quick check
    if len(images) > 5:
        images = random.sample(images, 5)

    transform = get_preprocessing(cfg['image_size'], cfg['image_size'])
    
    # Metric Accumulators
    metrics = {
        'roof_iou': [], 'roof_dice': [], 
        'edge_iou': [], 'edge_dice': [],
        'raw_edge_iou': [], 'raw_edge_dice': []
    }
    
    print(f"Running inference on {len(images)} images...")

    with torch.no_grad():
        for img_name in tqdm(images):
            img_path = os.path.join(input_image_dir, img_name)
            
            mask_name = os.path.splitext(img_name)[0] + ".png"
            gt_edge_path = os.path.join(gt_edge_dir, mask_name)
            gt_roof_path = os.path.join(gt_roof_dir, mask_name)
            
            original_img = cv2.imread(img_path)
            if original_img is None: continue
            orig_h, orig_w = original_img.shape[:2]
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Load GT Masks for Metrics
            gt_edge = cv2.imread(gt_edge_path, 0)
            gt_roof = cv2.imread(gt_roof_path, 0)
            
            # Create empty if missing
            if gt_edge is None: gt_edge = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if gt_roof is None: gt_roof = np.zeros((orig_h, orig_w), dtype=np.uint8)
            
            aug = transform(image=img_rgb)
            tensor_img = aug["image"].unsqueeze(0).to(device)

            # Model 1 Inference
            out1 = model1(tensor_img)
            prob1 = torch.sigmoid(out1)
            
            raw_edge_tensor = prob1[:, 0, :, :].unsqueeze(1)
            raw_roof_tensor = prob1[:, 1, :, :].unsqueeze(1)

            # Model 2 Inference
            out2 = model2(raw_edge_tensor)
            refined_edge_tensor = torch.sigmoid(out2)

            # Post-Processing
            def to_numpy_mask(tensor, threshold=0.5):
                mask = (tensor.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
                return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            roof_mask = to_numpy_mask(raw_roof_tensor, threshold=0.5)
            edge_mask_raw = to_numpy_mask(raw_edge_tensor, threshold=0.3) # Raw edge için threshold
            edge_mask_refined = to_numpy_mask(refined_edge_tensor, threshold=0.5)

            # --- Calculate Metrics for this image ---
            r_iou, r_dice = calculate_metrics(roof_mask, gt_roof)
            
            # Raw Edge Metrics (Model 1 output vs GT)
            raw_e_iou, raw_e_dice = calculate_metrics(edge_mask_raw, gt_edge)
            
            # Refined Edge Metrics (Model 2 output vs GT)
            e_iou, e_dice = calculate_metrics(edge_mask_refined, gt_edge)
            
            metrics['roof_iou'].append(r_iou)
            metrics['roof_dice'].append(r_dice)
            
            metrics['raw_edge_iou'].append(raw_e_iou)
            metrics['raw_edge_dice'].append(raw_e_dice)
            
            metrics['edge_iou'].append(e_iou)
            metrics['edge_dice'].append(e_dice)

            # Facet Extraction
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edge_mask_refined, kernel, iterations=1)
            facet_regions = cv2.subtract(roof_mask, dilated_edges)
            facet_regions = cv2.morphologyEx(facet_regions, cv2.MORPH_OPEN, kernel)

            # Contours and Polygons
            contours, _ = cv2.findContours(facet_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            final_vis = original_img.copy()
            colors = random_colors(len(contours))
            
            for idx, cnt in enumerate(contours):
                if cv2.contourArea(cnt) < 800:
                    continue
                
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                color = colors[idx]
                cv2.drawContours(final_vis, [approx], -1, color, -1)
                cv2.drawContours(final_vis, [approx], -1, (255, 255, 255), 2)

            # Create Overlays
            roof_overlay = original_img.copy()
            roof_overlay[roof_mask > 0] = (0, 0, 255) # Red for roof area
            roof_overlay = cv2.addWeighted(roof_overlay, 0.5, original_img, 0.5, 0)

            facet_overlay = cv2.addWeighted(final_vis, 0.6, original_img, 0.4, 0)

            # Combine Results: [Input] [Roof Overlay] [Facet Overlay]
            combined_result = np.hstack([
                add_title(original_img, "Input Image"),
                add_title(roof_overlay, f"Roof (Dice: {r_dice:.2f})"),
                add_title(facet_overlay, f"Facet (Edge Dice: {e_dice:.2f})")
            ])
            
            save_path = os.path.join(output_dir, f"result_{img_name}")
            cv2.imwrite(save_path, combined_result)

    # --- Generate Summary Report ---
    summary_path = os.path.join(output_dir, "metrics_summary.png")
    create_metrics_summary(metrics, summary_path)

if __name__ == "__main__":
    main()