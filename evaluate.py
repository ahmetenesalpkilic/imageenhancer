import torch
import cv2
import numpy as np
import os

from srcnn_model import SRCNN
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ------------------ AYARLAR ------------------
WEIGHTS_PATH = "srcnn_model_weights.pth"
SCALE_FACTOR = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ YCbCr (Y-channel) ------------------
def rgb2ycbcr(img):
    """
    img: RGB image, float32, [0,1]
    dönüş: Y channel, [0,1]
    """
    return (
        16.0 / 255.0
        + (65.481 * img[:, :, 0]
        + 128.553 * img[:, :, 1]
        + 24.966 * img[:, :, 2]) / 255.0
    )

# ------------------ EVALUATION ------------------
def evaluate_model():
    print(f"🔬 AKADEMİK TEST BAŞLIYOR ({DEVICE})\n")

    # ---- MODEL ----
    model = SRCNN().to(DEVICE)
    if not os.path.exists(WEIGHTS_PATH):
        print("❌ HATA: Model ağırlıkları bulunamadı.")
        return

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    # ---- LPIPS ----
    lpips_fn = LearnedPerceptualImagePatchSimilarity(
        net_type="alex"
    ).to(DEVICE)

    lr_dir = "lr_images"
    hr_dir = "hr_images"

    if not os.path.exists(lr_dir):
        print("❌ HATA: lr_images klasörü yok.")
        return

    files = os.listdir(lr_dir)
    if not files:
        print("❌ HATA: Klasör boş.")
        return

    # ---- METRİK LİSTELERİ ----
    avg_psnr_y = []
    avg_ssim_y = []
    avg_bicubic_psnr = []
    avg_lpips = []

    print(f"📂 Toplam {len(files)} resim test edilecek...")
    print("-" * 70)
    print(f"{'Dosya':<20} | {'SRCNN Y-PSNR':<14} | {'Bicubic Y-PSNR':<16} | {'LPIPS ↓':<8}")
    print("-" * 70)

    tested = 0

    for idx, filename in enumerate(files):
        lr_path = os.path.join(lr_dir, filename)
        base, ext = os.path.splitext(filename)
        hr_name = base.replace("_lr", "") + ext
        hr_path = os.path.join(hr_dir, hr_name)

        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)

        if lr_img is None or hr_img is None:
            continue

        h, w, _ = hr_img.shape

        # ---- BICUBIC UPSCALE ----
        lr_upscaled = cv2.resize(
            lr_img, (w, h), interpolation=cv2.INTER_CUBIC
        )

        # ---- MODEL INPUT ----
        inp_rgb = cv2.cvtColor(lr_upscaled, cv2.COLOR_BGR2RGB)
        inp_rgb = inp_rgb.astype(np.float32) / 255.0

        img_tensor = (
            torch.from_numpy(inp_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ---- INFERENCE ----
        with torch.no_grad():
            sr = model(img_tensor)

        sr = (
            sr.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)

        sr_bgr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        sr_bgr = sr_bgr[:h, :w]

        # ---- RGB ----
        hr_rgb = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
        bic_rgb = cv2.cvtColor(lr_upscaled, cv2.COLOR_BGR2RGB)

        # ---- Y CHANNEL ----
        hr_y = rgb2ycbcr(hr_rgb.astype(np.float32) / 255.0)
        sr_y = rgb2ycbcr(sr_rgb.astype(np.float32) / 255.0)
        bic_y = rgb2ycbcr(bic_rgb.astype(np.float32) / 255.0)

        # ---- BORDER CROP (SRCNN PAPER) ----
        shave = SCALE_FACTOR * 6
        hr_y = hr_y[shave:-shave, shave:-shave]
        sr_y = sr_y[shave:-shave, shave:-shave]
        bic_y = bic_y[shave:-shave, shave:-shave]

        # ---- PSNR / SSIM ----
        p_y = psnr(hr_y, sr_y, data_range=1.0)
        s_y = ssim(hr_y, sr_y, data_range=1.0)

        p_bic = psnr(hr_y, bic_y, data_range=1.0)

        avg_psnr_y.append(p_y)
        avg_ssim_y.append(s_y)
        avg_bicubic_psnr.append(p_bic)

        # ---- LPIPS (RGB, [-1,1]) ----
        sr_lp = torch.from_numpy(sr_rgb / 255.0).permute(2, 0, 1).unsqueeze(0)
        hr_lp = torch.from_numpy(hr_rgb / 255.0).permute(2, 0, 1).unsqueeze(0)

        sr_lp = sr_lp * 2 - 1
        hr_lp = hr_lp * 2 - 1

        sr_lp = sr_lp.to(DEVICE)
        hr_lp = hr_lp.to(DEVICE)

        with torch.no_grad():
            lp = lpips_fn(sr_lp, hr_lp)

        avg_lpips.append(lp.item())

        tested += 1

        print(
            f"{filename:<20} | {p_y:>6.2f} dB      | {p_bic:>6.2f} dB        | {lp.item():.4f}"
        )

        if idx == 0:
            cv2.imwrite("final_sr_output.png", sr_bgr)
            cv2.imwrite("final_bicubic.png", lr_upscaled)

    # ---- RAPOR ----
    print("-" * 70)
    print("\n" + "=" * 42)
    print("        📊 FİNAL SONUÇ RAPORU        ")
    print("=" * 42)

    print(f"Toplam Test Edilen Resim: {tested}")
    print("-" * 42)

    mean_psnr = np.mean(avg_psnr_y)
    mean_ssim = np.mean(avg_ssim_y)
    mean_bic = np.mean(avg_bicubic_psnr)
    mean_lpips = np.mean(avg_lpips)

    print(f"Ortalama Y-PSNR (SRCNN)   : {mean_psnr:.4f} dB")
    print(f"Ortalama Y-PSNR (Bicubic): {mean_bic:.4f} dB")
    print(f"Ortalama Y-SSIM (SRCNN)   : {mean_ssim:.4f}")
    print(f"Ortalama LPIPS (↓ daha iyi): {mean_lpips:.4f}")
    print("-" * 42)

    gain = mean_psnr - mean_bic
    if gain > 0:
        print(f"🚀 BAŞARILI! Ortalama {gain:.4f} dB PSNR iyileştirme.")
    else:
        print("⚠️ PSNR açısından bicubic gerisinde.")

# ------------------ RUN ------------------
if __name__ == "__main__":
    evaluate_model()
