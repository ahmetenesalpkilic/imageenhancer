import cv2
import os           #Klasör ve dosya islemleri
import glob         #Belirli uzantıya sahip dosyalari bulma
import numpy as np

#--Paramaetreler

HR_DIR="hr_images"      #yuksek cozunurluklu dosyalarimiz
LR_DIR="lr_images"      #olusturacagimiz düsük cozunurluklu dosyamiz
SCALE_FACTOR=4          #4 kat kucultme buyutme hedefimiz
ALLOWED_EXTENSIONS=['*.jpg','*.png','*.jpeg']

def prepare_dataset():
    #1- LR Klasorunu yoksa olustur: Ciktilari duzenli tutamak icin [2] 
    if not os.path.exists(LR_DIR):
        os.makedirs(LR_DIR)
        print(f"'{LR_DIR}' klasoru olusturuldu")

    #2- Islenecek dosyalar icin HR dosylarının listesini al
    hr_files = []

    for ext in ALLOWED_EXTENSIONS:
        # HR_DIR icindeki tum uzantilari topla
        hr_files.extend(glob.glob(os.path.join(HR_DIR, ext)))

    if not hr_files:
        print(f"Hata: '{HR_DIR}' klasorunde islenecek resim bulunamadi.")
        print("Lutfen hr_images klasorunuzu kontrol edin")
        return
    
    print(f"\n Islenilecek toplam HR goruntu sayisi {len(hr_files)}")

    #3. Her HR goruntuyu ısleyıp LR karsılıgını olustur
    for i, hr_path in enumerate(hr_files):
        base_name=os.path.basename(hr_path)   #dosya adini aliyor sadece (foto1.png)
        name, ext=os.path.splitext(base_name) #dosya adini ve uzanti olarak 2 ye boler

        hr_image=cv2.imread(hr_path)

        if hr_image is None:
            print(f"Uyari: {hr_path} okunamadi, atlaniyor")
            continue

        h, w, _ = hr_image.shape

        #yeni lr boyutlarini hesapla
        lr_w = w // SCALE_FACTOR
        lr_h = h // SCALE_FACTOR
        lr_dsize = (lr_w, lr_h)

        #bicubic downsampling (kucultme)
        #bu, modelin tersine cevirmeyi ogrenecegi bozulma turunu yaratir [1,2]
        lr_image=cv2.resize(
            hr_image,
            lr_dsize,
            interpolation=cv2.INTER_CUBIC
            #INTER_CUBIC → bicubic interpolation, hangi yontem oldugunu belirtir
        )
        # LR görüntüsünü kaydet
        output_path = os.path.join(LR_DIR, f"{name}_lr{ext}")
        #yeni kaydedilecek dosyanın yolunu oluşturuyor.
        cv2.imwrite(output_path, lr_image)

        print(f"  [{i+1}/{len(hr_files)}] {base_name} ({w}x{h}) -> {lr_w}x{lr_h} olarak kaydedildi.")
        #her bir resim islendiginde terminale bilgi yazdirir.
        
    print(f"\nVeri seti hazirligi tamamlandi. LR görüntüler '{LR_DIR}' klasöründe.")


if __name__ == "__main__": #main olarak calis,import edilince calismaz!
    prepare_dataset() 