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

# Custom Modules
from model import get_model

# SSL Fix
ssl._create_default_https_context = ssl._create_unverified_context

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_transforms(height, width):
    return A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def calculate_metrics(pred, target):
    """Basit IoU ve Dice hesaplama (Numpy array için)"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
    return iou, dice

def main():
    
    # 1. Ayarlar
    cfg = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Model Yükle
    model = get_model(cfg).to(device)
    checkpoint = torch.load(cfg['paths']['checkpoint_save_path'], map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Veri Yolları
    base_dir = cfg['paths']['base_dir']
    test_img_dir = os.path.join(base_dir, cfg['paths']['test_images'])
    test_edge_dir = os.path.join(base_dir, cfg['paths']['test_edge_masks'])
    test_roof_dir = os.path.join(base_dir, cfg['paths']['test_roof_masks'])

    # Resim Listesi
    all_images = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Rastgele 3 tane seç
    num_samples = 3
    if len(all_images) < num_samples: num_samples = len(all_images)
    selected_images = random.sample(all_images, num_samples)
    
    transform = get_transforms(cfg['hyperparameters']['image_height'], cfg['hyperparameters']['image_width'])
    
    # Metrik Toplayıcıları
    metrics = {"edge_iou": [], "edge_dice": [], "roof_iou": [], "roof_dice": []}
    visuals = []

    print(f"\nSeçilen {num_samples} görsel işleniyor...")

    with torch.no_grad():
        for img_name in selected_images:
            # Dosya Okuma
            img_path = os.path.join(test_img_dir, img_name)
            mask_name = os.path.splitext(img_name)[0] + ".png" # Maske uzantısı png varsayıldı
            
            original_img = cv2.imread(img_path)
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h, w = original_img.shape[:2]

            gt_edge = cv2.imread(os.path.join(test_edge_dir, mask_name), 0)
            gt_roof = cv2.imread(os.path.join(test_roof_dir, mask_name), 0)
            
            # Maske yoksa siyah oluştur
            if gt_edge is None: gt_edge = np.zeros((h, w), dtype=np.uint8)
            if gt_roof is None: gt_roof = np.zeros((h, w), dtype=np.uint8)

            # Binary yap (0-1)
            gt_edge_bin = (gt_edge > 127).astype(np.float32)
            gt_roof_bin = (gt_roof > 127).astype(np.float32)

            # Pre-process
            aug = transform(image=original_rgb)
            tensor_img = aug["image"].unsqueeze(0).to(device)

            # Predict
            logits = model(tensor_img)
            probs = torch.sigmoid(logits)
            
            # Post-process
            pred_edge = probs[0, 0].cpu().numpy()
            pred_roof = probs[0, 1].cpu().numpy()
            
            pred_edge_bin = (pred_edge > 0.5).astype(np.float32)
            pred_roof_bin = (pred_roof > 0.5).astype(np.float32)

            # Orijinal boyuta resize (Metrik hesabı için)
            pred_edge_resized = cv2.resize(pred_edge_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_roof_resized = cv2.resize(pred_roof_bin, (w, h), interpolation=cv2.INTER_NEAREST)

            # Metrik Hesapla
            e_iou, e_dice = calculate_metrics(pred_edge_resized, gt_edge_bin)
            r_iou, r_dice = calculate_metrics(pred_roof_resized, gt_roof_bin)
            
            metrics["edge_iou"].append(e_iou)
            metrics["edge_dice"].append(e_dice)
            metrics["roof_iou"].append(r_iou)
            metrics["roof_dice"].append(r_dice)

            # Görselleştirme için sakla (Normalize edilmiş resmi geri çevir)
            viz_img = tensor_img[0].permute(1, 2, 0).cpu().numpy()
            viz_img = viz_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            viz_img = np.clip(viz_img, 0, 1)
            
            visuals.append({
                "img": viz_img,
                "gt_edge": gt_edge, "pred_edge": pred_edge_bin,
                "gt_roof": gt_roof, "pred_roof": pred_roof_bin
            })

    # 4. Raporlama
    print("\n" + "="*40)
    print(f"HIZLI KONTROL RAPORU ({num_samples} Örnek)")
    print("="*40)
    print(f"Ortalama Edge Dice: {np.mean(metrics['edge_dice']):.4f}")
    print(f"Ortalama Edge IoU : {np.mean(metrics['edge_iou']):.4f}")
    print("-" * 40)
    print(f"Ortalama Roof Dice: {np.mean(metrics['roof_dice']):.4f}")
    print(f"Ortalama Roof IoU : {np.mean(metrics['roof_iou']):.4f}")
    print("="*40)

    # 5. Görselleştirme (Matplotlib)
    print("Görsel oluşturuluyor...")
    fig, axs = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1: axs = axs[np.newaxis, :]

    for i, viz in enumerate(visuals):
        # Input
        axs[i, 0].imshow(viz['img'])
        axs[i, 0].set_title("Input")
        axs[i, 0].axis("off")
        
        # GT Edge
        axs[i, 1].imshow(viz['gt_edge'], cmap='gray')
        axs[i, 1].set_title("GT Edge")
        axs[i, 1].axis("off")

        # Pred Edge
        axs[i, 2].imshow(viz['pred_edge'], cmap='gray')
        axs[i, 2].set_title(f"Pred Edge")
        axs[i, 2].axis("off")

        # GT Roof
        axs[i, 3].imshow(viz['gt_roof'], cmap='gray')
        axs[i, 3].set_title("GT Roof")
        axs[i, 3].axis("off")

        # Pred Roof
        axs[i, 4].imshow(viz['pred_roof'], cmap='gray')
        axs[i, 4].set_title(f"Pred Roof")
        axs[i, 4].axis("off")

    plt.tight_layout()
    plt.savefig("quick_check_result.png")
    print("✅ Sonuç görseli kaydedildi: quick_check_result.png")

if __name__ == "__main__":
    main()