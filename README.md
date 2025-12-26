# 🔍 YOLO Nesne Tespit Uygulaması (Object Detection)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv&logoColor=white)

Bu proje, **Bilgisayarlı Görü (Computer Vision)** dersi kapsamında geliştirilmiş, **YOLOv8** algoritmasını kullanarak görüntülerdeki nesneleri tespit eden bir masaüstü uygulamasıdır. Kullanıcı dostu arayüzü sayesinde herkesin kolayca nesne tespiti yapabilmesini sağlar.

---

## 📸 Ekran Görüntüsü

<img width="1919" height="1135" alt="Ekran görüntüsü 2025-12-25 014259" src="https://github.com/user-attachments/assets/12b9ee2d-7140-4d5f-b353-f72432794153" />
<img width="1919" height="1141" alt="Ekran görüntüsü 2025-12-25 025547" src="https://github.com/user-attachments/assets/9e754e90-85fa-4a7e-b87c-610a6cdaa4ac" />


## ✨ Özellikler

* **🔍 Gerçek Zamanlı Tespit:** YOLOv8n (Nano) modeli ile hızlı ve yüksek doğruluklu nesne tespiti.
* **📂 Kolay Resim Yükleme:** JPG, PNG, JPEG formatlarını destekler.
* **📊 Detaylı Analiz:**
    * Nesnelerin sınıf isimleri (İnsan, Araba, Kedi vb.)
    * Güven skorları (Confidence Score)
    * Tespit edilen toplam nesne sayısı
* **🎨 Görselleştirme:** Her nesne sınıfı için farklı renkte sınırlayıcı kutular (Bounding Box).
* **💻 Kullanıcı Arayüzü:** Python Tkinter ile geliştirilmiş modern arayüz.

---

## 🛠️ Kurulum

Bu projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1. Gereksinimler
Projenin çalışması için bilgisayarınızda Python yüklü olmalıdır. Gerekli kütüphaneleri yüklemek için terminal veya komut satırına şu kodu yapıştırın:

```bash
pip install ultralytics opencv-python numpy pillow
