import torch
import cv2
import numpy as np
import os
from srcnn_model import SRCNN
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- Ayarlar ---
WEIGHTS_PATH = "srcnn_model_weights.pth"
SCALE_FACTOR = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rgb2ycbcr(img):
    """
    Görüntüyü RGB'den YCbCr formatına çevirir ve sadece Y (Luminance) kanalını döndürür.
    Formül: Y = 65.481 * R + 128.553 * G + 24.966 * B + 16
    (Matlab standartlarına uygun dönüşüm - SRCNN makalelerinde bu kullanılır)
    """
    y = 16. + (65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2])
    return y / 255.0

def evaluate_model():
    print(f"🔬 AKADEMİK TEST BAŞLIYOR ({DEVICE})...\n")

    # 1. Modeli Yükle
    model = SRCNN().to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    else:
        print("❌ HATA: Model ağırlıkları bulunamadı.")
        return
    model.eval()

    lr_dir = "lr_images"
    hr_dir = "hr_images"

    if not os.path.exists(lr_dir):
        print("❌ HATA: lr_images klasörü yok.")
        return

    files = os.listdir(lr_dir)
    if not files:
        print("❌ HATA: Klasör boş.")
        return

    # İstatistikleri tutacak listeler
    avg_psnr_rgb, avg_ssim_rgb = [], []
    avg_psnr_y, avg_ssim_y = [], []
    
    avg_bicubic_psnr, avg_bicubic_ssim = [], []

    print(f"📂 Toplam {len(files)} resim test edilecek...")
    print("-" * 60)
    print(f"{'Dosya':<20} | {'SRCNN (Y) PSNR':<15} | {'Bicubic (Y) PSNR':<15}")
    print("-" * 60)

    for idx, filename in enumerate(files):
        # Dosya Yolları
        lr_path = os.path.join(lr_dir, filename)
        base_name, ext = os.path.splitext(filename)
        hr_name = base_name.replace("_lr", "") + ext
        hr_path = os.path.join(hr_dir, hr_name)

        # Okuma
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)

        if lr_img is None or hr_img is None:
            continue

        h, w, _ = hr_img.shape
        
        # Bicubic Upscale (Model Girdisi)
        lr_upscaled = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)

        # Tensor Hazırlığı
        img_input = cv2.cvtColor(lr_upscaled, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # Model Tahmini
        with torch.no_grad():
            output = model(img_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()

        # Kırpma / Boyut Düzeltme
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        # BGR dönüşümü (OpenCV formatı)
        sr_img_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Boyut eşitleme (Crop)
        out_h, out_w, _ = sr_img_bgr.shape
        sr_img_bgr = sr_img_bgr[:h, :w]
        
        # --- METRİK HESAPLAMA ---
        
        # 1. RGB Metrikleri (İnsan gözü için genel referans)
        hr_rgb = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        sr_rgb = cv2.cvtColor(sr_img_bgr, cv2.COLOR_BGR2RGB)
        bic_rgb = cv2.cvtColor(lr_upscaled, cv2.COLOR_BGR2RGB)
        
        avg_psnr_rgb.append(psnr(hr_rgb, sr_rgb, data_range=255))
        avg_ssim_rgb.append(ssim(hr_rgb, sr_rgb, channel_axis=2, data_range=255))

        # 2. Y-Channel Metrikleri (Akademik Standart)
        # Görüntüleri 0-1 aralığına çekip Y kanalını alıyoruz
        hr_y = rgb2ycbcr(hr_rgb.astype(np.float32) / 255.0)
        sr_y = rgb2ycbcr(sr_rgb.astype(np.float32) / 255.0)
        bic_y = rgb2ycbcr(bic_rgb.astype(np.float32) / 255.0)
        
        # Y-PSNR Hesapla (data_range=1.0 çünkü float 0-1 arası)
        p_y = psnr(hr_y, sr_y, data_range=1.0)
        s_y = ssim(hr_y, sr_y, data_range=1.0)
        
        p_bic_y = psnr(hr_y, bic_y, data_range=1.0)
        s_bic_y = ssim(hr_y, bic_y, data_range=1.0)

        avg_psnr_y.append(p_y)
        avg_ssim_y.append(s_y)
        avg_bicubic_psnr.append(p_bic_y)

        # Her 5 resimde bir veya sonuncuda yazdır
        print(f"{filename:<20} | {p_y:.2f} dB        | {p_bic_y:.2f} dB")

        # Örnek görsel kaydet (Sadece ilk resmi)
        if idx == 0:
            cv2.imwrite("final_sr_output.png", sr_img_bgr)
            cv2.imwrite("final_bicubic.png", lr_upscaled)

    print("-" * 60)
    print("\n" + "="*40)
    print("       📊 FİNAL SONUÇ RAPORU       ")
    print("="*40)
    
    mean_psnr = np.mean(avg_psnr_y)
    mean_ssim = np.mean(avg_ssim_y)
    mean_bicubic = np.mean(avg_bicubic_psnr)
    
    print(f"Toplam Test Edilen Resim: {len(files)}")
    print("-" * 40)
    print(f"Ortalama Y-PSNR (SRCNN)  : {mean_psnr:.4f} dB")
    print(f"Ortalama Y-PSNR (Bicubic): {mean_bicubic:.4f} dB")
    print(f"Ortalama Y-SSIM (SRCNN)  : {mean_ssim:.4f}")
    print("-" * 40)
    
    gain = mean_psnr - mean_bicubic
    if gain > 0:
        print(f"🚀 BAŞARILI! Model ortalamada {gain:.4f} dB iyileştirme sağladı.")
    else:
        print(f"⚠️ HENÜZ DEĞİL. Ortalama performans klasik yöntemin gerisinde.")

if __name__ == "__main__":
    evaluate_model()