# train.py dosyası

from sr_dataset import SRDataset # Kendi veri setimiz
from srcnn_model import SRCNN     # Kendi modelimiz
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os

#Egitim parametreleri
HR_DIR="hr_images"
LR_DIR="lr_images"
BATCH_SIZE=16       #Bir adimda islenecek yama sayisi
NUM_EPOCHS=10       #Veri setinin bir kez baştan sona işlenceği
LEARNING_RATE=0.001 
# GPU/CPU kontrolU: Hiz icin GPU (CUDA) varsa onu kullan
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    #1. Veri yukleyiciyi hazirla
    train_dataset = SRDataset(hr_dir=HR_DIR,lr_dir=LR_DIR)

    #Dataloader: Veri yamalarini toplu halde hazirlar (Batching)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,           #Her epochta veriyi karistir
        num_workers=0
    )

    #2. Modeli ve Bilesenleri tanimla
    model=SRCNN().to(DEVICE)

    # Kayip Fonksiyonu: Mean Squared Error (MSE). PSNR'yi maksimize etmeyi hedefler [3, 4]
    criterion=nn.MSELoss()

    # Optimizasyon: Adam algoritmasi, agirliklari güncellemek icin kullanilir
    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
    
    print(f"\nEgitim {DEVICE} uzerinde basliyor...")

    #3.Egitim dongusu
    for epoch in range(1,NUM_EPOCHS +1):
        model.train()   #Modeli egitim moduna al
        running_loss=0.0

        #DataLoader'dan LR ve HR yamalari (batchler) alinir
        for i,(lr_patches, hr_patches) in enumerate(train_loader):

            #Veriyi cihaza (GPU/CPU) aktar
            lr_patches=lr_patches.to(DEVICE)
            hr_patches=hr_patches.to(DEVICE)

            #Optimizasyon sifirlamasi
            optimizer.zero_grad()

            #ileri besleme (Forward pass) LR yamasini modele ver, SR ciktisini al
            sr_output=model(lr_patches)
            
            # Kayip Hesaplama (Loss Calculation): SR ciktisi ile Gercek HR hedefi karsilastirilir
            loss = criterion(sr_output, hr_patches)

            # Geriye yayilim (Backward Pass): Hatanin ag boyunca yayilmasi
            loss.backward()

            #Agirlik guncelleme
            optimizer.step()

            running_loss += loss.item()

        # Her epoch sonunda ortalama kaybi yazdir
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Ortalama Kayip (Loss): {avg_loss:.6f}")   

    print("\n Egitim Tamamlandi!")

    #Egitilmis model agirliklarini kaydet (Daha sonra test etmek icin)

    torch.save(model.state_dict(), "srcnn_model_weights.pth")
    print("Model ağırlıkları 'srcnn_model_weights.pth' olarak kaydedildi.")

if __name__ == "__main__":
    train_model()                  

