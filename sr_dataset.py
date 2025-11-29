# sr_dataset.py dosyası

import torch
from torch.utils.data import Dataset # PyTorch'ta veri yönetimi icin sinif
import cv2
import glob
import os
import random
import numpy as np

# --- Modelin eğitim parametreleri ---
PATCH_SIZE = 128    # Modelin bir seferde işleyeceği yama boyutu (128x128) [1]
SCALE_FACTOR = 4

class SR_Dataset(Dataset):
    """
    Super Resolution projesi icin LR ve HR görüntü ciftlerini yöneten PyTorch sinifi
    Rastgele yama cikarma (patching) ve normalizasyon islemlerini yapar.
    """
    def __init__(self, hr_dir, lr_dir, patch_size=PATCH_SIZE, scale_factor=SCALE_FACTOR):
        # 1. HR ve LR dosya yollarini eslestirir
        self.hr_paths= sorted(glob.glob(os.path.join(hr_dir,'*.*')))
        self.lr_paths= sorted(glob.glob(os.path.join(lr_dir,'*.*')))

        if(len(self.hr_paths)!=len(self.lr_paths)):
            print("HATA: HR ve LR klasorlerindeki  dosya sayisi esit degil!")
            raise ValueError
        
        self.patch_size=patch_size
        self.scale_factor=scale_factor

        print(f"Dataset hazir. Toplam {len(self.hr_paths)} cift goruntu bulundu")

    def __len__(self):
        #Toplam goruntu cifti sayisi
        return len(self.hr_paths)
    def __getitem__(self,idx):
        #Bir index (idx) verildiginde , o cifte ait HR ve LR yamasını dondurur

        hr_image = cv2.imread(self.hr_paths[idx])
        lr_image = cv2.imread(self.lr_paths[idx])

        #BGR (OpenCV) --> RGB (DL standardi) donusumu
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # 1. Rastgele Yama Alma (Patch Extraction)
        h, w, _ = hr_image.shape
        hr_patch_size = self.patch_size

        # Rastgele baslangic koordinatlarini belirle
        i = random.randint(0, h - hr_patch_size)
        j = random.randint(0, w - hr_patch_size)

        # HR yamasini kes (Hedef)
        hr_patch = hr_image[i:i + hr_patch_size, j:j + hr_patch_size]

        # LR yamasini kes (Girdi - Koordinatlari 4'e boluyoruz, 4x olcek faktörü nedeniyle)
        lr_patch_size = hr_patch_size // self.scale_factor
        lr_patch = lr_image[i // self.scale_factor : (i + hr_patch_size) // self.scale_factor, 
                            j // self.scale_factor : (j + hr_patch_size) // self.scale_factor]
        
        #2- Onisleme  (preprocessing)

        #2a Normalizasyon : 0-255 --> 0-1 araligina getir [2]
        hr_patch = hr_patch.astype(np.float32) /255.0
        lr_patch = lr_patch.astype(np.float32) /255.0

        #2b Boyut Sirasini Degistirme: Numpy (HxWxK) --> PyTORCH (KxHxW)
        # PyTorch, Kanal bilgisinin (K) ilk boyutta olmasini bekler.
        hr_patch = torch.from_numpy(hr_patch).permute(2, 0, 1)
        lr_patch = torch.from_numpy(lr_patch).permute(2, 0, 1)
        #tensör formatini deep learning standartina cevirmek
        
        return lr_patch, hr_patch 
        

 