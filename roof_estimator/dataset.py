import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class RoofDataset(Dataset):
    def __init__(self, image_dir, edge_mask_dir, area_mask_dir, transform=None):
        self.image_dir = image_dir
        self.edge_mask_dir = edge_mask_dir
        self.area_mask_dir = area_mask_dir
        self.transform = transform
        
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        
        self.images = sorted([
            f for f in os.listdir(image_dir) 
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, index):
        img_name = self.images[index]
        
        img_path = os.path.join(self.image_dir, img_name)
        edge_path = os.path.join(self.edge_mask_dir, os.path.splitext(img_name)[0] + ".png")
        area_path = os.path.join(self.area_mask_dir, os.path.splitext(img_name)[0] + ".png")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_edge = cv2.imread(edge_path, 0)
        mask_area = cv2.imread(area_path, 0)
        
        if mask_edge is None:
             raise FileNotFoundError(f"Edge Mask not found: {edge_path}")
        if mask_area is None:
             raise FileNotFoundError(f"Area Mask not found: {area_path}")

        mask_edge = mask_edge.astype(np.float32)
        mask_edge[mask_edge > 0.0] = 1.0 
        
        mask_area = mask_area.astype(np.float32)
        mask_area[mask_area > 0.0] = 1.0


        if self.transform is not None:
            augmentations = self.transform(image=image, masks=[mask_edge, mask_area])
            image = augmentations["image"]
            combined_mask = augmentations["masks"] 

        return image, combined_mask
