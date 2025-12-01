import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class MaskRefinementDataset(Dataset):
    def __init__(self, input_mask_dir, label_mask_dir, transform=None):
        self.input_mask_dir = input_mask_dir
        self.label_mask_dir = label_mask_dir
        self.transform = transform
        
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        
        self.input_masks = sorted([f for f in os.listdir(input_mask_dir) if os.path.splitext(f)[1].lower() in valid_extensions])
        self.label_masks = sorted([f for f in os.listdir(label_mask_dir) if os.path.splitext(f)[1].lower() in valid_extensions])

        if len(self.input_masks) != len(self.label_masks):
            print(f"WARNING: Input ({len(self.input_masks)}) and Label ({len(self.label_masks)}) numbers are not equal")

    def __len__(self):
        return len(self.input_masks)

    def __getitem__(self, index):
        inp_name = self.input_masks[index]
        inp_path = os.path.join(self.input_mask_dir, inp_name)
        
        input_mask = cv2.imread(inp_path, 0) 
        if input_mask is None:
            raise FileNotFoundError(f"Input mask not found: {inp_path}")
        
        input_mask = input_mask.astype(np.float32) / 255.0

        lbl_name = self.label_masks[index] 
        lbl_path = os.path.join(self.label_mask_dir, lbl_name)
        
        label_mask = cv2.imread(lbl_path, 0)
        if label_mask is None:
            raise FileNotFoundError(f"Label mask not found: {lbl_path}")
            
        label_mask = label_mask.astype(np.float32)
        label_mask[label_mask > 0.0] = 1.0 

        if self.transform is not None:
            augmentations = self.transform(image=input_mask, mask=label_mask)
            input_mask = augmentations["image"] 
            label_mask = augmentations["mask"]  

        if input_mask.ndim == 2:
            input_mask = input_mask.unsqueeze(0)
            
        return input_mask, label_mask