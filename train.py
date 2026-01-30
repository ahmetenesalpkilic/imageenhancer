import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time

from dataset import SRDataset
from model import SRCNN

# =====================
# AYARLAR
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 100          # DOKUNMA
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_WORKERS = 8
PIN_MEMORY = True

# =====================
# DATASET
# =====================
train_dataset = SRDataset(split="train")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=True
)

# =====================
# MODEL
# =====================
model = SRCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scaler = GradScaler()

# =====================
# TRAIN
# =====================
start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for lr_img, hr_img in train_loader:
        lr_img = lr_img.to(DEVICE, non_blocking=True)
        hr_img = hr_img.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            sr_img = model(lr_img)
            loss = criterion(sr_img, hr_img)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # LOG + CHECKPOINT
    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - start
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Loss: {avg_loss:.6f} | Süre: {elapsed:.1f}s")

        torch.save(model.state_dict(), "srcnn_checkpoint.pth")

# =====================
# FINAL MODEL
# =====================
torch.save(model.state_dict(), "srcnn_final.pth")
print("✅ Eğitim tamamlandı, final model kaydedildi.")
