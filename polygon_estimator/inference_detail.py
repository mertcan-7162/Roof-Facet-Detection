import os
import cv2
import yaml
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class InferenceDataset(Dataset):
    def __init__(self, image_dir, input_mask_dir, label_mask_dir, transform=None):
        self.image_dir = image_dir
        self.input_mask_dir = input_mask_dir
        self.label_mask_dir = label_mask_dir
        self.transform = transform
        
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        
        # Dosya listelerini oluştur
        self.images = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_extensions])
        self.input_masks = sorted([f for f in os.listdir(input_mask_dir) if os.path.splitext(f)[1].lower() in valid_extensions])
        self.label_masks = sorted([f for f in os.listdir(label_mask_dir) if os.path.splitext(f)[1].lower() in valid_extensions])

        # Basit eşleşme kontrolü (Sadece uyarı)
        if len(self.images) != len(self.input_masks):
            print(f"UYARI: Resim sayısı ({len(self.images)}) ile Maske sayısı ({len(self.input_masks)}) uyuşmuyor!")

    def __len__(self):
        return len(self.input_masks) # Maske sayısı kadar işlem yapalım

    def __getitem__(self, index):
        # Dosya isimlerinin eşleştiğini varsayıyoruz (sorted yapıldığı için)
        mask_name = self.input_masks[index]
        
        # 1. Orijinal Resim (SADECE GÖRSELLEŞTİRME İÇİN)
        # Resim ismini maske isminden türetmeye çalışıyoruz (uzantı farkı olabilir)
        # Basitlik adına aynı isimde olduğunu varsayıyoruz veya index kullanıyoruz
        img_name = self.images[index] 
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None: # Hata yönetimi
            # Resim yoksa siyah dön
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Input Mask (MODEL GİRDİSİ)
        inp_mask_path = os.path.join(self.input_mask_dir, mask_name)
        input_mask = cv2.imread(inp_mask_path, 0)
        input_mask = input_mask.astype(np.float32) / 255.0 

        # 3. Label Mask (KIYASLAMA İÇİN)
        lbl_mask_path = os.path.join(self.label_mask_dir, self.label_masks[index])
        label_mask = cv2.imread(lbl_mask_path, 0)
        label_mask = label_mask.astype(np.float32)
        label_mask[label_mask > 0.0] = 1.0 

        # Transform
        if self.transform is not None:
            # Albumentations'a resmi de veriyoruz ki boyutlar eşitlensin
            augmentations = self.transform(image=image, mask=label_mask, input_mask=input_mask)
            
            image_tensor = augmentations["image"]      
            label_tensor = augmentations["mask"]
            input_tensor = augmentations["input_mask"]
        
        # Model 1 kanal (1, H, W) bekliyor, Albumentations (H, W) verebilir
        if input_tensor.ndim == 2:
            input_tensor = input_tensor.unsqueeze(0)
            
        return input_tensor, label_tensor, image_tensor, img_name

def get_inference_transforms(height, width):
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
        additional_targets={'input_mask': 'mask'}
    )

def tensor_to_img_numpy(tensor):
    """(C, H, W) Tensor'u (H, W, C) Numpy uint8'e çevirir."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)

def add_title(image, title):
    # Grayscale ise 3 kanala çevir
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1: # (H, W, 1) durumu
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    h, w = image.shape[:2]
    canvas = np.zeros((h + 40, w, 3), dtype=np.uint8)
    canvas[40:, :] = image
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas

def main():
    # 1. CONFIG YÜKLEME
    cfg = load_config("/Users/mert/Documents/PROJECTS/SOLARVIS/train_trials/final_code/polygon_estimator/config.yaml")
    
    # Cihaz ayarı
    if cfg['hyperparameters']['device'] == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = cfg['hyperparameters']['device']
    print(f"Device: {device}")

    # Yollar
    base_dir = cfg['paths']['base_dir']
    
    # Test için kullanacağımız klasörleri config'den alıyoruz.
    # NOT: Config dosyanızda 'test_images' yolu yoksa manuel ekleyebilirsiniz 
    # veya burada olduğu gibi varsayılan bir yol tanımlayabiliriz.
    
    # Burası önemli: Orijinal resimlerin olduğu klasör (Config'de yoksa manuel düzeltilmeli)
    # Genelde base_dir'in bir üstünde 'data/test_images' gibi durur.
    test_image_dir = os.path.join(base_dir, cfg['paths']['test_input_masks']) 
    if not os.path.exists(test_image_dir):
        # Eğer yukarıdaki yol yoksa config'deki bir yolu deneyelim veya uyarı verelim
        print(f"UYARI: Test resim klasörü bulunamadı: {test_image_dir}")
        # Fallback (Eğitim verisi görselleştirmesi için)
        # test_image_dir = ...
    
    test_input_mask_dir = os.path.join(base_dir, cfg['paths']['test_input_masks'])
    test_label_mask_dir = os.path.join(base_dir, cfg['paths']['test_label_masks'])
    
    # Çıktı Klasörü
    output_dir = os.path.join(base_dir, "polygon_complete_inference_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # 2. MODEL YÜKLEME
    print("Model yükleniyor...")
    model = smp.UnetPlusPlus(
        encoder_name=cfg['model']['encoder_name'],        
        encoder_weights=None, # Hazır ağırlık yüklemiyoruz, kendi checkpointimizi yükleyeceğiz
        in_channels=cfg['model']['in_channels'],
        classes=cfg['model']['classes'],                      
        activation=None, 
    ).to(device)
    
    model_path = cfg['paths']['checkpoint_save_path']
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"HATA: '{model_path}' bulunamadı. Lütfen config dosyasını kontrol et.")
        return

    model.eval()

    # 3. VERİ SETİ HAZIRLIĞI
    h, w = cfg['hyperparameters']['image_height'], cfg['hyperparameters']['image_width']
    
    ds = InferenceDataset(
        image_dir=test_image_dir,
        input_mask_dir=test_input_mask_dir,
        label_mask_dir=test_label_mask_dir,
        transform=get_inference_transforms(h, w)
    )
    
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    print(f"Toplam {len(ds)} resim işlenecek...")

    # 4. INFERENCE LOOP
    with torch.no_grad():
        for batch_idx, (input_mask, target_mask, image_tensor, img_name) in enumerate(tqdm(loader)):
            input_mask = input_mask.to(device)
            
            # --- MODEL TAHMİNİ ---
            logits = model(input_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # --- GÖRSELLEŞTİRME İŞLEMLERİ ---
            
            # 1. Orijinal Resim (Tensor -> Numpy)
            original_img = tensor_to_img_numpy(image_tensor[0])
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR) # OpenCV için BGR

            # 2. Input Mask (Bozuk olan)
            input_mask_img = (input_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)

            # 3. Ground Truth
            target_img = (target_mask[0].cpu().numpy() * 255).astype(np.uint8)

            # 4. Model Output (Tamamlanmış)
            pred_img = (preds[0, 0].cpu().numpy() * 255).astype(np.uint8)

            # 5. Dilation (Kalınlaştırma - Opsiyonel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            dilated_pred = cv2.dilate(pred_img, kernel, iterations=1)

            # Başlık ekle
            img1 = add_title(original_img, "Original Image")
            img2 = add_title(input_mask_img, "Input (Broken)")
            img3 = add_title(target_img, "Ground Truth")
            img4 = add_title(pred_img, "Refined Output")
            img5 = add_title(dilated_pred, "Dilation Applied")

            # Yan yana birleştir (Horizontal Stack)
            combined = np.hstack((img1, img2, img3, img4, img5))

            # Kaydet
            save_name = f"vis_{img_name[0]}"
            # Dosya uzantısı yoksa .png ekle
            if not save_name.lower().endswith(('.png', '.jpg')):
                save_name += ".png"
                
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, combined)

    print(f"✅ İşlem tamamlandı! Sonuçlar '{output_dir}' klasöründe.")

if __name__ == "__main__":
    main()