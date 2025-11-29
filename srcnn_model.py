# srcnn_model.py dosyası

import torch
import torch.nn as nn

# --- Model Parametreleri ---
N_CHANNELS = 3    
N_FEATURES_1 = 64 
N_FEATURES_2 = 32

class SRCNN(nn.Module):
    """
    PyTorch ile SRCNN (Super-Resolution Convolutional Neural Network) Mimarisi.
    """
    def __init__(self):
        super(SRCNN,self).__init__()

        #1. Asama ozellik cıkarma
        # Girdi: 3 (LR) | Cikti: 64 Ozellik  haritasi | Kernel: 9x9, Padding: 4 (Boyut koruma icin)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=N_CHANNELS, out_channels=N_FEATURES_1, kernel_size=9, padding=4), #bu görüntünün üzerinden gezip 9x9 filtrelerle detayları cikarir
            nn.ReLU()
        )

        # 2. Asama: Haritalama
        # Girdi: 64 ozellik | Cikti: 32 özellik haritasi | Kernel: 5x5, Padding: 2
        self.mapping = nn.Sequential(
            nn.Conv2d(in_channels=N_FEATURES_1, out_channels=N_FEATURES_2, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # 3. Asama: Yeniden Yapilandirma (Cikti)
        # Girdi: 32 özellik | Cikti: 3 (Tahmin Edilen HR Görüntü) | Kernel: 5x5, Padding: 2
        self.reconstruction = nn.Conv2d(
            in_channels=N_FEATURES_2,
            out_channels=N_CHANNELS,
            kernel_size=5,
            padding=2
        )

    def forward(self,x):
        """" Veri akisini tanimlar. Girdi (LR), cikti (SR) """
        x = self.feature_extractor(x)
        x = self.mapping(x) 
        x = self.reconstruction(x)
        return x
           

