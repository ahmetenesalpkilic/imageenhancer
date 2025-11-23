# ImageEnhancer

## English

**ImageEnhancer** is a Super Resolution project that enhances low-resolution (LR) images to high-resolution (HR) images using Bicubic downsampling and Deep Learning models. The project is designed to create 2x–4x magnified versions of images while preserving details and textures.

### Features
- Generate LR-HR pairs for training
- Prepare datasets with customizable scale factors (e.g., 2x, 4x)
- Easily extendable for different Deep Learning architectures
- Fast preprocessing with OpenCV and Python

### Requirements
- Python 3.8+
- OpenCV
- NumPy
- scikit-image (for image metrics and evaluations)

### How to Use
1. Place your high-resolution images in `hr_images/` folder (this folder is ignored by GitHub to avoid large files).
2. Run `data_prep.py` to generate low-resolution images in `lr_images/`.
3. Train your Super Resolution model using the LR-HR pairs.

### Notes
- `hr_images/` folder is excluded from GitHub because it may contain large datasets.
- Adjust `SCALE_FACTOR` in `data_prep.py` according to your training needs.

---

## Türkçe

**ImageEnhancer**, düşük çözünürlüklü (LR) görüntüleri Bicubic küçültme ve Derin Öğrenme modelleri kullanarak yüksek çözünürlüklü (HR) görüntülere yükselten bir Süper Çözünürlük projesidir. Proje, görüntüleri 2x–4x büyütürken detay ve dokuları korumayı hedefler.

### Özellikler
- Eğitim için LR-HR çiftleri oluşturur
- Ölçek faktörü özelleştirilebilir (örn. 2x, 4x)
- Farklı Derin Öğrenme mimarileri ile kolayca genişletilebilir
- OpenCV ve Python ile hızlı veri ön işleme

### Gereksinimler
- Python 3.8+
- OpenCV
- NumPy
- scikit-image (görüntü metrikleri ve değerlendirme için)

### Kullanım
1. Yüksek çözünürlüklü görüntülerinizi `hr_images/` klasörüne koyun (bu klasör GitHub’a pushlanmaz, büyük dosyalar içerir).
2. `data_prep.py` dosyasını çalıştırarak düşük çözünürlüklü görüntüleri `lr_images/` klasöründe oluşturun.
3. LR-HR çiftlerini kullanarak Süper Çözünürlük modelinizi eğitin.

### Notlar
- `hr_images/` klasörü GitHub’a dahil edilmez, büyük veri setleri içerebilir.
- `data_prep.py` içindeki `SCALE_FACTOR` değerini eğitim ihtiyaçlarınıza göre ayarlayın.
