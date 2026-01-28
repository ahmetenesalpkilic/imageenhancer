from sr_dataset import SRCNNDataset
from srcnn_model import SRCNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# --- 1. Parametreler ---
HR_DIR = "hr_images"
LR_DIR = "lr_images"
BATCH_SIZE = 16
NUM_EPOCHS = 100       # ÖNEMLİ: Modelin öğrenmesi için en az 100 epoch önerilir 
LEARNING_RATE = 0.0001 # Daha hassas öğrenme için hız düşürüldü

# Cihaz Seçimi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ GPU AKTİF: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU BULUNAMADI, CPU ile devam ediliyor (Yavaş olabilir)...")

def train_model():
    print("\n--- 1. Hazırlık Aşaması ---")
    
    # 2. Dataset ve DataLoader
    try:
        train_dataset = SRCNNDataset(
            hr_dir=HR_DIR,
            lr_dir=LR_DIR,
            patch_size=33, # SRCNN orijinal kağıt değeri [3]
            scale_factor=4
        )
        
        # num_workers=0: Windows'ta çoklu işlem hatalarını önlemek için kritiktir.
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,  
            drop_last=True
        )
        print(f"Dataset Yüklendi. Toplam Resim Çifti: {len(train_dataset)}")
    except Exception as e:
        print(f"❌ HATA: Dataset yüklenemedi: {e}")
        return

    # 3. Model, Kayıp Fonksiyonu ve Optimizasyon
    model = SRCNN().to(DEVICE)
    
    # MSE Loss: PSNR değerini doğrudan artırmayı amaçlar [4, 5]
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- 2. Eğitim Başlıyor (Hedef: {NUM_EPOCHS} Epoch) ---")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for i, (lr_patches, hr_patches) in enumerate(train_loader):
            lr_patches = lr_patches.to(DEVICE)
            hr_patches = hr_patches.to(DEVICE)

            # Gradyan sıfırlama ve İleri besleme
            optimizer.zero_grad()
            sr_output = model(lr_patches)
            
            # Kayıp hesaplama ve Geriye yayılım
            loss = criterion(sr_output, hr_patches)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Her 20 batch'te bir durum göster
            if i % 20 == 0:
                print(".", end="", flush=True)

        avg_loss = running_loss / len(train_loader)
        
        # Her 10 epoch'ta bir detaylı durum yazdır
        if epoch % 10 == 0 or epoch == 1:
            elapsed_time = time.time() - start_time
            print(f"\n🚀 Epoch | Ortalama Kayıp: {avg_loss:.6f} | Süre: {elapsed_time:.1f}sn")
            
            # Ara ağırlıkları kaydet (Çökme ihtimaline karşı yedek)
            torch.save(model.state_dict(), "srcnn_checkpoint.pth")

    # 4. Final Kayıt
    print("\n🎉 EĞİTİM TAMAMLANDI!")
    torch.save(model.state_dict(), "srcnn_model_weights.pth")
    print(f"Model ağırlıkları 'srcnn_model_weights.pth' olarak kaydedildi.")

if __name__ == "__main__":
    train_model()