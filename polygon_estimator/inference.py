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

# Model mimarisini model.py dosyasından çekiyoruz
from model import get_model

# SSL Sertifika hatası düzeltmesi
ssl._create_default_https_context = ssl._create_unverified_context

def load_config(config_path="/Users/mert/Documents/PROJECTS/SOLARVIS/train_trials/final_code/polygon_estimator/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_transforms(height, width):
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
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

def calculate_metrics(pred, target):
    """
    Tekil bir resim için IoU ve Dice hesaplar.
    Girdi: (H, W) boyutunda Binary Numpy Array (0 veya 1)
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
    
    return iou, dice

def main():
    print("--- Hızlı Model Kontrolü Başlatılıyor ---")
    
    # 1. Ayarları Yükle
    cfg = load_config("/Users/mert/Documents/PROJECTS/SOLARVIS/train_trials/final_code/polygon_estimator/config.yaml")
    
    # Cihaz Seçimi
    if cfg['hyperparameters']['device'] == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = cfg['hyperparameters']['device']
    print(f"Device: {device}")

    # 2. Modeli Yükle
    print("Model yükleniyor...")
    model = get_model(cfg).to(device)
    
    checkpoint_path = cfg['paths']['checkpoint_save_path']
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Checkpoint bazen 'state_dict' anahtarı içinde olur
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"HATA: Model yüklenemedi! Yol: {checkpoint_path}\nHata: {e}")
        return

    model.eval()

    base_dir = cfg['paths']['base_dir']
    test_img_dir = os.path.join(base_dir, cfg['paths']['test_images'])
    test_edge_dir = os.path.join(base_dir, cfg['paths']['test_edge_masks'])
    test_roof_dir = os.path.join(base_dir, cfg['paths']['test_roof_masks'])

    # Test klasöründeki resimleri listele
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}
    all_images = sorted([f for f in os.listdir(test_img_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    if not all_images:
        print(f"UYARI: {test_img_dir} klasöründe resim bulunamadı!")
        return

    # Rastgele 3 tane seç
    num_samples = 3
    selected_images = random.sample(all_images, min(len(all_images), num_samples))
    
    # Preprocessing
    h, w = cfg['hyperparameters']['image_height'], cfg['hyperparameters']['image_width']
    transform = get_transforms(h, w)
    
    # İstatistikler için listeler
    metrics = {"edge_iou": [], "edge_dice": [], "roof_iou": [], "roof_dice": []}
    visuals = []

    print(f"\nSeçilen {len(selected_images)} görsel işleniyor...")

    with torch.no_grad():
        for img_name in selected_images:
            # --- Dosya Okuma ---
            img_path = os.path.join(test_img_dir, img_name)
            mask_name = os.path.splitext(img_name)[0] + ".png"
            
            # Görüntü
            original_img = cv2.imread(img_path)
            if original_img is None: continue
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = original_img.shape[:2]

            # GT Maskeler
            gt_edge = cv2.imread(os.path.join(test_edge_dir, mask_name), 0)
            gt_roof = cv2.imread(os.path.join(test_roof_dir, mask_name), 0)
            
            # Maske yoksa siyah oluştur (Hata vermesin diye)
            if gt_edge is None: gt_edge = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if gt_roof is None: gt_roof = np.zeros((orig_h, orig_w), dtype=np.uint8)

            # Binary'e çevir (0-1)
            gt_edge_bin = (gt_edge > 127).astype(np.float32)
            gt_roof_bin = (gt_roof > 127).astype(np.float32)

            # --- Model Tahmini ---
            aug = transform(image=original_rgb)
            tensor_img = aug["image"].unsqueeze(0).to(device)

            logits = model(tensor_img)
            probs = torch.sigmoid(logits)
            
            # Çıktıları ayır (Varsayım: Ch0=Edge, Ch1=Roof)
            # Eğer eğitimde tam tersiyse buradaki indeksleri [0, 1] -> [0, 0] değiştir.
            pred_edge_prob = probs[0, 0, :, :].cpu().numpy()
            pred_roof_prob = probs[0, 1, :, :].cpu().numpy()
            
            # Thresholding
            pred_edge_bin = (pred_edge_prob > 0.5).astype(np.float32)
            pred_roof_bin = (pred_roof_prob > 0.5).astype(np.float32)

            # Orijinal boyuta geri döndür (Metrik hesabı adil olsun diye)
            # Çünkü model 640x640'a pad etmiş olabilir.
            pred_edge_resized = cv2.resize(pred_edge_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            pred_roof_resized = cv2.resize(pred_roof_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # --- Metrik Hesapla ---
            e_iou, e_dice = calculate_metrics(pred_edge_resized, gt_edge_bin)
            r_iou, r_dice = calculate_metrics(pred_roof_resized, gt_roof_bin)
            
            metrics["edge_iou"].append(e_iou)
            metrics["edge_dice"].append(e_dice)
            metrics["roof_iou"].append(r_iou)
            metrics["roof_dice"].append(r_dice)

            # --- Görselleştirme Hazırlığı ---
            # Tensor'u geri resme çevir (Normalize işlemini geri al)
            viz_img = tensor_img[0].permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            viz_img = viz_img * std + mean
            viz_img = np.clip(viz_img, 0, 1)
            
            visuals.append({
                "img": viz_img,
                "gt_edge": gt_edge, 
                "pred_edge": pred_edge_bin, # Görselde 640'lık hali kalsın, daha net görünür
                "gt_roof": gt_roof, 
                "pred_roof": pred_roof_bin
            })

    # 4. Sonuçları Raporla
    print("\n" + "="*45)
    print(f" HIZLI KONTROL SONUÇLARI ({len(metrics['edge_dice'])} Örnek)")
    print("="*45)
    print(f"Edge Detection:")
    print(f"  - Dice Score : {np.mean(metrics['edge_dice']):.4f}")
    print(f"  - IoU Score  : {np.mean(metrics['edge_iou']):.4f}")
    print("-" * 45)
    print(f"Roof Segmentation:")
    print(f"  - Dice Score : {np.mean(metrics['roof_dice']):.4f}")
    print(f"  - IoU Score  : {np.mean(metrics['roof_iou']):.4f}")
    print("="*45)

    # 5. Görsel Kaydet (Matplotlib)
    if visuals:
        print("Görsel oluşturuluyor...")
        rows = len(visuals)
        cols = 5 # Input | GT Edge | Pred Edge | GT Roof | Pred Roof
        
        fig, axs = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        if rows == 1: axs = axs[np.newaxis, :] # Boyut hatasını önle

        for i, viz in enumerate(visuals):
            # Input
            axs[i, 0].imshow(viz['img'])
            axs[i, 0].set_title("Input Image")
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
        plt.savefig("quick_check_result.png", dpi=150)
        print("✅ Görsel kaydedildi: quick_check_result.png")
    
    print("\nİşlem Tamamlandı.")

if __name__ == "__main__":
    main()