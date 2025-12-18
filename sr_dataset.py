import torch
from torch.utils.data import Dataset
import cv2
import glob
import os
import random
import numpy as np

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=128, scale_factor=4):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.*')))
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.*')))
        
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        
        if len(self.hr_paths) != len(self.lr_paths):
            raise ValueError("HATA: HR ve LR klasörlerindeki dosya sayısı eşit değil!")
        
        print(f"Dataset hazır. Toplam {len(self.hr_paths)} çift yama işlenecek.")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # 1. Görüntüleri oku ve RGB'ye çevir
        hr_image = cv2.imread(self.hr_paths[idx])
        lr_image = cv2.imread(self.lr_paths[idx])
        
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # 2. Koordinatları Belirle (Hizalama hatasını önlemek için scale_factor katı)
        h, w, _ = hr_image.shape
        i = random.randrange(0, h - self.patch_size + 1, self.scale_factor)
        j = random.randrange(0, w - self.patch_size + 1, self.scale_factor)

        # 3. HR Yamayı Kes
        hr_patch = hr_image[i:i + self.patch_size, j:j + self.patch_size]
        
        # 4. LR Yamayı Kes
        lr_i, lr_j = i // self.scale_factor, j // self.scale_factor
        lr_p_size = self.patch_size // self.scale_factor
        lr_patch = lr_image[lr_i:lr_i + lr_p_size, lr_j:lr_j + lr_p_size]

        # --- KRİTİK SRCNN ADIMI: Bicubic Büyütme ---
        # Model, HR boyutunda (bulanık) bir girdi beklediği için 32x32'yi 128x128'e çıkarıyoruz
        lr_patch = cv2.resize(
            lr_patch, 
            (self.patch_size, self.patch_size), 
            interpolation=cv2.INTER_CUBIC
        )

        # 5. Veri Artırma (Data Augmentation)
        if random.random() > 0.5:
            hr_patch = np.ascontiguousarray(hr_patch[::-1, :, :])
            lr_patch = np.ascontiguousarray(lr_patch[::-1, :, :])
        if random.random() > 0.5:
            hr_patch = np.ascontiguousarray(hr_patch[:, ::-1, :])
            lr_patch = np.ascontiguousarray(lr_patch[:, ::-1, :])

        # 6. Normalizasyon ve Tensor Dönüşümü
        hr_patch = torch.from_numpy(hr_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        lr_patch = torch.from_numpy(lr_patch.astype(np.float32) / 255.0).permute(2, 0, 1)

        return lr_patch, hr_patch