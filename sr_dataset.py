import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import glob
import random

class SRCNNDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=33, scale_factor=4):
        # NOT: scale_factor varsayılan olarak 4 yapıldı.
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.*")))
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.*")))
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        
        # Basit bir dosya sayısı kontrolü
        # (Sayılar eşit değilse data_prep aşamasında bazı dosyalar bozuk diye atlanmış olabilir, 
        # bu durumda sadece eşleşenleri almak daha güvenlidir ama şimdilik assert bırakıyoruz)
        assert len(self.hr_paths) == len(self.lr_paths), "HR ve LR dosya sayıları eşit değil!"
        if len(self.hr_paths) == 0:
            raise RuntimeError(f"Klasörde resim bulunamadı: {hr_dir}")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # 🟢 GÜVENLİ DÖNGÜ: Bozuk resim gelirse 10 kereye kadar başkasını dener
        for _ in range(10):
            hr_path = self.hr_paths[idx]
            lr_path = self.lr_paths[idx]

            # 1. Okuma
            hr_image = cv2.imread(hr_path)
            lr_image = cv2.imread(lr_path)

            if hr_image is None or lr_image is None:
                idx = random.randint(0, len(self.hr_paths) - 1)
                continue

            # BGR -> RGB
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

            # 2. Boyut Kontrolü
            lr_h, lr_w, _ = lr_image.shape
            lr_patch_size = self.patch_size // self.scale_factor
            
            # Scale Factor 4 olunca patch'ler çok küçülür (33 // 4 = 8px).
            # Resim bundan bile küçükse atla.
            if (lr_h < lr_patch_size or lr_w < lr_patch_size or
                hr_image.shape[0] < self.patch_size or hr_image.shape[1] < self.patch_size):
                idx = random.randint(0, len(self.hr_paths) - 1)
                continue

            # 3. Rastgele Crop Koordinatları
            try:
                i = random.randint(0, lr_h - lr_patch_size)
                j = random.randint(0, lr_w - lr_patch_size)
            except ValueError:
                idx = random.randint(0, len(self.hr_paths) - 1)
                continue

            # LR Patch Kes
            lr_patch = lr_image[i:i + lr_patch_size, j:j + lr_patch_size]

            # HR Patch Kes (Koordinatları scale ile çarp)
            hr_i = i * self.scale_factor
            hr_j = j * self.scale_factor
            hr_patch = hr_image[hr_i:hr_i + self.patch_size, hr_j:hr_j + self.patch_size]

            # 4. Patch Boyut Doğrulama
            if hr_patch.shape[0] != self.patch_size or hr_patch.shape[1] != self.patch_size:
                idx = random.randint(0, len(self.hr_paths) - 1)
                continue

            # 5. SRCNN Ön İşleme: LR Patch'i Bicubic ile HR boyutuna büyüt
            # (Scale 4 olduğu için 8x8'lik resim 33x33'e bulanık şekilde büyütülür)
            lr_patch = cv2.resize(lr_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

            # 6. Augmentation
            if random.random() > 0.5:
                lr_patch = np.ascontiguousarray(lr_patch[:, ::-1, :])
                hr_patch = np.ascontiguousarray(hr_patch[:, ::-1, :])

            if random.random() > 0.5:
                lr_patch = np.ascontiguousarray(lr_patch[::-1, :, :])
                hr_patch = np.ascontiguousarray(hr_patch[::-1, :, :])

            # 7. Normalizasyon ve Tensor
            lr_patch = lr_patch.astype(np.float32) / 255.0
            hr_patch = hr_patch.astype(np.float32) / 255.0

            lr_patch = torch.from_numpy(lr_patch).permute(2, 0, 1)
            hr_patch = torch.from_numpy(hr_patch).permute(2, 0, 1)

            return lr_patch, hr_patch

        raise RuntimeError(f"HATA: {idx} indexli resim için uygun patch bulunamadı.")