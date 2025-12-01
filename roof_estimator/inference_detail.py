import os
import cv2
import yaml
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import ssl

# Custom Modules (Aynı klasörde olmalılar)
from model import get_model

# SSL Fix
ssl._create_default_https_context = ssl._create_unverified_context

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_inference_transforms(height, width):

    return A.Compose([
        A.PadIfNeeded(
            min_height=height, 
            min_width=width, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            mask_value=0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

def create_dirs(base_output_dir):
    sub_dirs = {
        "edges_raw": os.path.join(base_output_dir, "model_out", "pred_edges"),
        "roofs_raw": os.path.join(base_output_dir, "model_out", "pred_roofs"),
        "edge_grid": os.path.join(base_output_dir, "edge_grid_compare"),
        "roof_grid": os.path.join(base_output_dir, "roof_grid_compare"),
        "edge_overlay": os.path.join(base_output_dir, "edge_overlay_compare"),
        "roof_overlay": os.path.join(base_output_dir, "roof_overlay_compare"),
    }
    
    for d in sub_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    return sub_dirs

def create_overlay(image, mask, color=(0, 255, 0)):
    mask_bool = mask > 127
    overlay = image.copy()
    overlay[mask_bool] = color
    return overlay

def create_grid_image(original, my_pred, gt, title_pred="Prediction", title_gt="Ground Truth"):

    h, w, _ = original.shape
    
    def to_3ch(gray_mask):
        if len(gray_mask.shape) == 2:
            return cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        return gray_mask

    my_pred_bgr = to_3ch(my_pred)
    gt_bgr = to_3ch(gt)
    
    # Boyut eşitleme (Garantilemek için)
    my_pred_bgr = cv2.resize(my_pred_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    gt_bgr = cv2.resize(gt_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

    # Yazı ekleme
    font_scale = 1.0
    thickness = 2
    cv2.putText(original, "Original", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness)
    cv2.putText(gt_bgr, title_gt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
    cv2.putText(my_pred_bgr, title_pred, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)

    return np.hstack((original, gt_bgr, my_pred_bgr))

def main():
    cfg = load_config("roof_estimator/config.yaml")
    
    if cfg['hyperparameters']['device'] == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    else:
        device = cfg['hyperparameters']['device']
    
    print(f"--- Inference Started on {device} ---")

    base_dir = cfg['paths']['base_dir']
    test_img_dir = os.path.join(base_dir, cfg['paths']['test_images'])
    test_edge_dir = os.path.join(base_dir, cfg['paths']['test_edge_masks'])
    test_roof_dir = os.path.join(base_dir, cfg['paths']['test_roof_masks'])
    model_path = cfg['paths']['checkpoint_save_path']
    
    output_base_dir = os.path.join(base_dir, "..", "roof_estimator/detail_roof_detect_inference_results") 

    out_dirs = create_dirs(output_base_dir)
    print(f"Results will be saved to: {output_base_dir}")

    print("Loading Model...")
    model = get_model(cfg).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    h, w = cfg['hyperparameters']['image_height'], cfg['hyperparameters']['image_width']
    transform = get_inference_transforms(h, w)
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    image_list = sorted([f for f in os.listdir(test_img_dir) if os.path.splitext(f)[1].lower() in valid_exts])

    for img_name in tqdm(image_list, desc="Processing Images"):
        try:
            img_path = os.path.join(test_img_dir, img_name)
            original_img = cv2.imread(img_path)
            if original_img is None: continue
            
            orig_h, orig_w = original_img.shape[:2]
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # GT Maskeleri Okuma (Varsa) - png uzantılı varsayıyoruz
            mask_name = os.path.splitext(img_name)[0] + ".png"
            
            gt_edge = cv2.imread(os.path.join(test_edge_dir, mask_name), 0)
            if gt_edge is None: gt_edge = np.zeros((orig_h, orig_w), dtype=np.uint8)
            
            gt_roof = cv2.imread(os.path.join(test_roof_dir, mask_name), 0)
            if gt_roof is None: gt_roof = np.zeros((orig_h, orig_w), dtype=np.uint8)

            # --- Pre-processing ---
            augmented = transform(image=img_rgb) 
            tensor_img = augmented["image"].unsqueeze(0).to(device) # (1, 3, H, W)

            # --- Prediction ---
            with torch.no_grad():
                logits = model(tensor_img)
                probs = torch.sigmoid(logits)
                
                # Channel 0 -> Edge 
                # Channel 1 -> Roof 
                edge_pred_tensor = probs[:, 0, :, :]
                roof_pred_tensor = probs[:, 1, :, :]

            # --- Post-processing ---
            edge_pred_np = edge_pred_tensor.squeeze().cpu().numpy()
            roof_pred_np = roof_pred_tensor.squeeze().cpu().numpy()

            # Threshold
            edge_mask = (edge_pred_np > 0.5).astype(np.uint8) * 255
            roof_mask = (roof_pred_np > 0.5).astype(np.uint8) * 255

            edge_mask = cv2.resize(edge_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            roof_mask = cv2.resize(roof_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


            # RAW OUTPUTS
            cv2.imwrite(os.path.join(out_dirs["edges_raw"], mask_name), edge_mask)
            cv2.imwrite(os.path.join(out_dirs["roofs_raw"], mask_name), roof_mask)

            # GRIDS (Comparison)
            # Edge Grid
            edge_grid = create_grid_image(original_img, edge_mask, gt_edge, title_pred="Pred Edge", title_gt="GT Edge")
            cv2.imwrite(os.path.join(out_dirs["edge_grid"], mask_name), edge_grid)
            
            # Roof Grid
            roof_grid = create_grid_image(original_img, roof_mask, gt_roof, title_pred="Pred Roof", title_gt="GT Roof")
            cv2.imwrite(os.path.join(out_dirs["roof_grid"], mask_name), roof_grid)

            # OVERLAYS 
            # Edge Overlay
            edge_ov = create_overlay(original_img, edge_mask, color=(0, 255, 0))
            cv2.imwrite(os.path.join(out_dirs["edge_overlay"], mask_name), edge_ov)
            
            # Roof Overlay 
            roof_ov = create_overlay(original_img, roof_mask, color=(0, 0, 255)) 
            cv2.imwrite(os.path.join(out_dirs["roof_overlay"], mask_name), roof_ov)

        except Exception as e:
            print(f"Skipping {img_name}: {e}")
            continue

    print("\n Inference completed successfully!")

if __name__ == "__main__":
    main()