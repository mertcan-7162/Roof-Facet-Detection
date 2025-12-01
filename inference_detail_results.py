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

# Fix for SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# --- LOAD CONFIGURATION ---
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# --- MODEL LOADER ---
def load_model(encoder, in_channels, classes, checkpoint_path, device):
    print(f"Loading Model: {encoder} (In: {in_channels}, Out: {classes})...")
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
        print("Weights loaded.")
    except Exception as e:
        print(f"Error loading weights from {checkpoint_path}: {e}")
        exit(1)
        
    model.to(device)
    model.eval()
    return model

# --- PREPROCESSING ---
def get_preprocessing(height, width):
    return A.Compose([
        A.PadIfNeeded(
            min_height=height, 
            min_width=width, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

# --- VISUALIZATION HELPERS ---
def add_title(img, text, color=(255, 255, 255), bg_color=(0, 0, 0)):
    h, w = img.shape[:2]
    # Convert grayscale to BGR if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    canvas = np.zeros((h + 40, w, 3), dtype=np.uint8)
    canvas[:] = bg_color 
    canvas[40:, :] = img
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas

def create_colored_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """Creates a colored overlay from a grayscale mask."""
    binary_mask = mask > 127
    overlay = image.copy()
    overlay[binary_mask] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def extract_polygons(roof_mask, edge_mask, original_img, morph_kernel=3, epsilon_ratio=0.01):
    """
    Extracts, cleans, simplifies, and draws facet polygons.
    """
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)

    # 1. Morphological closing and dilation on edges
    closed_edge = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    dilated_edge = cv2.dilate(closed_edge, kernel, iterations=1)

    # 2. Subtraction: Roof - Edge to isolate facets
    facets_raw = cv2.subtract(roof_mask, dilated_edge)

    # 3. Morphological opening to remove noise
    facets_clean = cv2.morphologyEx(facets_raw, cv2.MORPH_OPEN, kernel)

    # 4. Contour Extraction
    contours, _ = cv2.findContours(facets_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = original_img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800: # Ignore small noise
            continue

        # Douglas-Peucker Algorithm (Vector Simplification)
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Generate random color
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        
        # Draw polygon and vertices
        cv2.drawContours(result_img, [approx], -1, color, 2) 
        for point in approx:
            cv2.circle(result_img, tuple(point[0]), 3, (0, 0, 255), -1)

    return result_img


def main():
    cfg = load_config()
    
    if cfg.get('device', 'auto') == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = cfg['device']

    # Model 1: Input RGB -> Output Roof + Edge
    model1 = load_model(
        cfg['model1']['encoder'], 
        cfg['model1']['in_channels'], 
        cfg['model1']['classes'], 
        cfg['model1']['path'], 
        device
    )
    
    # Model 2: Input Raw Edge -> Output Refined Edge
    model2 = load_model(
        cfg['model2']['encoder'], 
        cfg['model2']['in_channels'], 
        cfg['model2']['classes'], 
        cfg['model2']['path'], 
        device
    )

    input_image_dir = os.path.join(cfg['paths']['base_dir'], cfg['paths']['test_images'])
    output_dir = cfg['paths']['output_dir']
    
    # Detailed output directories
    dirs = {
        "detail_inference_results": os.path.join(output_dir, "detail_inference_results"),
        "raw_refined": os.path.join(output_dir, "detail_inference_results/1_model_output"),
        "collage": os.path.join(output_dir, "detail_inference_results/2_visualization_collage"),
        "overlay": os.path.join(output_dir, "detail_inference_results/3_overlay_comparison"),
        "facets": os.path.join(output_dir, "detail_inference_results/4_facet_polygons")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    valid_exts = {'.png', '.jpg', '.jpeg', '.tif'}
    images = sorted([f for f in os.listdir(input_image_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    transform = get_preprocessing(cfg['image_size'], cfg['image_size'])
    
    print(f"Processing {len(images)} images...")

    # 4. Inference Loop
    with torch.no_grad():
        for img_name in tqdm(images):
            try:
                img_path = os.path.join(input_image_dir, img_name)
                
                original_img = cv2.imread(img_path)
                if original_img is None: continue
                orig_h, orig_w = original_img.shape[:2]
                img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                aug = transform(image=img_rgb)
                tensor_img = aug["image"].unsqueeze(0).to(device)

                out1 = model1(tensor_img)
                prob1 = torch.sigmoid(out1)
                
                raw_edge_tensor = prob1[:, 0, :, :].unsqueeze(1)
                raw_roof_tensor = prob1[:, 1, :, :].unsqueeze(1)

                out2 = model2(raw_edge_tensor)
                refined_edge_tensor = torch.sigmoid(out2)

                def to_numpy_mask(tensor, threshold=0.5):
                    mask = (tensor.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
                    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                roof_mask = to_numpy_mask(raw_roof_tensor, threshold=0.5)
                edge_mask_raw = to_numpy_mask(raw_edge_tensor, threshold=0.3)
                edge_mask_refined = to_numpy_mask(refined_edge_tensor, threshold=0.5)


                # A. Raw Refined Output
                cv2.imwrite(os.path.join(dirs["raw_refined"], img_name), edge_mask_refined)

                # B. Collage [Original | Roof | Raw Edge | Refined Edge]
                def to_bgr(gray): return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                collage = np.hstack([
                    add_title(original_img, "Input Image"),
                    add_title(to_bgr(roof_mask), "M1: Roof Area"),
                    add_title(to_bgr(edge_mask_raw), "M1: Raw Edge"),
                    add_title(to_bgr(edge_mask_refined), "M2: Refined Edge")
                ])
                # Resize collage if too large
                scale = 0.8
                h_c, w_c = collage.shape[:2]
                collage = cv2.resize(collage, (int(w_c*scale), int(h_c*scale)))
                cv2.imwrite(os.path.join(dirs["collage"], f"collage_{img_name}"), collage)

                # C. Overlay Comparison
                # Left: Original + Roof (Red)
                overlay_left = create_colored_overlay(original_img, roof_mask, color=(0, 0, 255), alpha=0.4)
                overlay_left = add_title(overlay_left, "Overlay: Roof Area")
                
                # Right: Original + Refined Edge (Green)
                overlay_right = create_colored_overlay(original_img, edge_mask_refined, color=(0, 255, 0), alpha=0.6)
                overlay_right = add_title(overlay_right, "Overlay: Refined Edge")
                
                overlay_combined = np.hstack((overlay_left, overlay_right))
                cv2.imwrite(os.path.join(dirs["overlay"], f"overlay_{img_name}"), overlay_combined)

                # D. Facet Polygons
                # Get parameters from config
                morph = cfg.get('post_processing', {}).get('morph_kernel', 3)
                epsilon = cfg.get('post_processing', {}).get('poly_epsilon', 0.01)
                
                poly_img = extract_polygons(roof_mask, edge_mask_refined, original_img, morph, epsilon)
                cv2.imwrite(os.path.join(dirs["facets"], f"poly_{img_name}"), poly_img)

            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue

    print(f"\n Processing complete. Results saved to '{output_dir}'.")

if __name__ == "__main__":
    main()