import torch
import cv2
import numpy as np
import os
import sys

# Dosya yapına göre bu dosyanın yanına srcnn_model.py koymalısın
from srcnn_model import SRCNN 
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ================== AYARLAR ==================
WEIGHTS_PATH = "srcnn_model_weights.pth"
LR_DIR = "lr_images"
HR_DIR = "hr_images"
OUTPUT_DIR = "outputs"

SCALE_FACTOR = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== YARDIMCI FONKSİYONLAR ==================
def find_hr_image(lr_filename, hr_dir):
    """
    LR dosya adına göre HR dosyayı bulur.
    Örn: 'resim_lr.png' -> 'resim.png' olarak arar.
    """
    base, ext = os.path.splitext(lr_filename)
    
    # Dosya isminde _lr eki varsa temizle
    if base.endswith("_lr"):
        hr_name = base[:-3] + ext
    else:
        hr_name = lr_filename
        
    hr_path = os.path.join(hr_dir, hr_name)
    
    if os.path.exists(hr_path):
        return hr_path
    return None

def rgb2ycbcr(img):
    """
    Akademik standartlarda Y (Luminance) kanalı dönüşümü.
    img: RGB image, float32, [0,1]
    return: Y channel, [0,1]
    """
    return (
        16.0 / 255.0
        + (65.481 * img[:, :, 0]
        + 128.553 * img[:, :, 1]
        + 24.966 * img[:, :, 2]) / 255.0
    )

# ================== EVALUATION ==================
def evaluate_model():
    print(f"\n🔬 AKADEMİK TEST BAŞLIYOR ({DEVICE})\n")

    # ---- 1. KONTROLLER ----
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ HATA: '{WEIGHTS_PATH}' dosyası bulunamadı.")
        return
    if not os.path.exists(LR_DIR) or not os.path.exists(HR_DIR):
        print("❌ HATA: 'lr_images' veya 'hr_images' klasörleri eksik.")
        return

    # ---- 2. MODELİ YÜKLE ----
    model = SRCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return
    model.eval()

    # ---- 3. LPIPS METRİĞİ ----
    print("⏳ LPIPS yükleniyor (ilk seferde biraz sürebilir)...")
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(DEVICE)
    lpips_fn.eval()

    files = sorted(os.listdir(LR_DIR))
    if not files:
        print("❌ HATA: lr_images klasörü boş.")
        return

    # Metrik Listeleri
    avg_psnr_y, avg_ssim_y = [], []
    avg_bicubic_psnr, avg_lpips = [], []

    print(f"\n📂 Toplam {len(files)} resim test edilecek...")
    print("-" * 78)
    print(f"{'Dosya':<20} | {'SRCNN Y-PSNR':<14} | {'Bicubic Y-PSNR':<16} | {'LPIPS ↓':<8}")
    print("-" * 78)

    tested = 0

    for idx, filename in enumerate(files):
        lr_path = os.path.join(LR_DIR, filename)
        hr_path = find_hr_image(filename, HR_DIR)

        if hr_path is None:
            print(f"⚠️ HR karşılığı bulunamadı, atlandı: {filename}")
            continue

        # Resimleri Oku
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)

        if lr_img is None or hr_img is None:
            continue

        h, w, _ = hr_img.shape

        # ---- BICUBIC UPSCALE ----
        lr_upscaled = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)

        # ---- MODEL INPUT HAZIRLAMA ----
        # BGR -> RGB ve [0-1] Float dönüşümü
        inp_rgb = cv2.cvtColor(lr_upscaled, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # !!!! DÜZELTİLEN KISIM BURASI !!!!
        img_tensor = (
            torch.from_numpy(inp_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()  # <--- HATA ÇÖZÜMÜ: Double'dan Float'a çeviriyoruz
            .to(DEVICE)
        )

        # ---- TAHMİN (INFERENCE) ----
        with torch.no_grad():
            sr = model(img_tensor)

        # Tensor -> Numpy dönüşümü
        sr = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)
        sr_bgr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        # ---- METRİK HESAPLAMA İÇİN HAZIRLIK ----
        hr_rgb = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
        bic_rgb = cv2.cvtColor(lr_upscaled, cv2.COLOR_BGR2RGB)

        # Y Kanalına Dönüşüm (Akademik Standart)
        hr_y = rgb2ycbcr(hr_rgb.astype(np.float32) / 255.0)
        sr_y = rgb2ycbcr(sr_rgb.astype(np.float32) / 255.0)
        bic_y = rgb2ycbcr(bic_rgb.astype(np.float32) / 255.0)

        # Kenarları Kırpma (Shave)
        shave = SCALE_FACTOR * 6
        if h > 2 * shave and w > 2 * shave: # Resim çok küçükse hata vermesin
            hr_y = hr_y[shave:-shave, shave:-shave]
            sr_y = sr_y[shave:-shave, shave:-shave]
            bic_y = bic_y[shave:-shave, shave:-shave]

        # PSNR ve SSIM
        p_y = psnr(hr_y, sr_y, data_range=1.0)
        s_y = ssim(hr_y, sr_y, data_range=1.0)
        p_bic = psnr(hr_y, bic_y, data_range=1.0)

        avg_psnr_y.append(p_y)
        avg_ssim_y.append(s_y)
        avg_bicubic_psnr.append(p_bic)

        # LPIPS Hesaplama ([-1, 1] aralığına çekilir)
        # Burası da float() olmalı
        sr_lp = (torch.from_numpy(sr_rgb).permute(2,0,1).unsqueeze(0).float()/255.0 * 2 - 1).to(DEVICE)
        hr_lp = (torch.from_numpy(hr_rgb).permute(2,0,1).unsqueeze(0).float()/255.0 * 2 - 1).to(DEVICE)

        with torch.no_grad():
            lp = lpips_fn(sr_lp, hr_lp)
        avg_lpips.append(lp.item())

        tested += 1
        print(f"{filename:<20} | {p_y:>6.2f} dB      | {p_bic:>6.2f} dB       | {lp.item():.4f}")

        # İlk resmi örnek olarak kaydet
        if idx == 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, "final_sr.png"), sr_bgr)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "final_bicubic.png"), lr_upscaled)

    # ---- FİNAL RAPORU ----
    if tested > 0:
        print("-" * 78)
        print("\n" + "=" * 42)
        print("         📊 FİNAL SONUÇ RAPORU         ")
        print("=" * 42)
        print(f"Toplam Test Edilen: {tested}")
        print("-" * 42)
        print(f"Ortalama Y-PSNR (SRCNN)    : {np.mean(avg_psnr_y):.4f} dB")
        print(f"Ortalama Y-PSNR (Bicubic)  : {np.mean(avg_bicubic_psnr):.4f} dB")
        print(f"Ortalama Y-SSIM (SRCNN)    : {np.mean(avg_ssim_y):.4f}")
        print(f"Ortalama LPIPS (Düşük iyi) : {np.mean(avg_lpips):.4f}")
        
        gain = np.mean(avg_psnr_y) - np.mean(avg_bicubic_psnr)
        print("-" * 42)
        if gain > 0:
            print(f"🚀 BAŞARILI! Model Bicubic'e göre +{gain:.4f} dB daha iyi.")
        else:
            print("⚠️ Model henüz Bicubic'i geçemedi.")
    else:
        print("\n❌ Hiçbir resim test edilemedi. Dosya isimlerini ve klasörleri kontrol et.")

if __name__ == "__main__":
    evaluate_model()