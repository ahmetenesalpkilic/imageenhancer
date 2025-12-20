from sr_dataset import SRCNNDataset
from srcnn_model import SRCNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import time

# --- Parametreler ---
HR_DIR = "hr_images"
LR_DIR = "lr_images"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Cihaz Seçimi
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ GPU BULUNDU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ GPU BULUNAMADI, CPU kullanılıyor...")

def train_model():
    print("\n--- 1. Hazırlık Aşaması ---")
    print("Dataset dosyaları taranıyor...")
    
    # Dataset Yükleme (Scale Factor 4)
    try:
        train_dataset = SRCNNDataset(
            hr_dir=HR_DIR,
            lr_dir=LR_DIR,
            patch_size=33,
            scale_factor=4
        )
        print(f"Dataset Başarılı! Toplam Resim: {len(train_dataset)}")
    except Exception as e:
        print(f"❌ Dataset Hatası: {e}")
        return

    # DataLoader
    print("DataLoader hazırlanıyor (num_workers=0)...")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,    # Windows için kritik ayar (0 olmalı)
        drop_last=True
    )
    print("DataLoader Hazır.")

    # Model
    print(f"Model {DEVICE} cihazına aktarılıyor...")
    model = SRCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model Hazır.")

    print("\n--- 2. Eğitim Başlıyor ---")
    print("Lütfen bekleyin, ilk veri paketi (batch) hazırlanıyor...")
    
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        # İlk batch'in ne zaman geldiğini görmek için sayaç
        for i, (lr_patches, hr_patches) in enumerate(train_loader):
            if i == 0:
                print(f"⚡ İlk Batch Geldi! (Süre: {time.time() - start_time:.1f} sn)")
                print("GPU işlemeye başladı...")

            lr_patches = lr_patches.to(DEVICE)
            hr_patches = hr_patches.to(DEVICE)

            optimizer.zero_grad()
            sr_output = model(lr_patches)
            loss = criterion(sr_output, hr_patches)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Kullanıcıya çalıştığını hissettirmek için her 5 batch'te bir nokta koy
            if i % 10 == 0:
                print(".", end="", flush=True)

        avg_loss = running_loss / len(train_loader)
        print(f"\n✅ Epoch [{epoch}/{NUM_EPOCHS}] Tamamlandı - Loss: {avg_loss:.6f}")

    print("\n🎉 Eğitim Bitti!")
    torch.save(model.state_dict(), "srcnn_model_weights.pth")
    print("Model kaydedildi.")

if __name__ == "__main__":
    train_model()