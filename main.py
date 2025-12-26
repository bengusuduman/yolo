"""
ÖDEV 5 - ÜÇÜNCÜ GÖREV: YOLO Nesne Tespit Uygulaması

Bu uygulama, YOLO (You Only Look Once) algoritması kullanarak görüntülerde
nesne tespiti yapar. En az 5 farklı nesne tespit edilebilir.

YOLO ALGORİTMASI HAKKINDA AÇIKLAMA:
===================================

YOLO Nedir?
-----------
YOLO (You Only Look Once), gerçek zamanlı nesne tespiti için geliştirilmiş
devrimsel bir derin öğrenme algoritmasıdır. Geleneksel yöntemlerin aksine,
YOLO görüntüyü tek bir ileri geçişte (single forward pass) işleyerek
hem nesne konumlarını hem de sınıflarını aynı anda tahmin eder.

Algoritma Çalışma Prensibi:
---------------------------
1. GİRİŞ: Görüntü S×S boyutunda bir ızgaraya (grid) bölünür
2. TAHMİN: Her ızgara hücresi B adet sınırlayıcı kutu (bounding box) ve
   güven skoru (confidence score) tahmin eder
3. SINIFLANDIRMA: Her hücre için C sınıf olasılığı hesaplanır
4. NMS: Non-Maximum Suppression ile çakışan kutular elenir
5. ÇIKIŞ: Final tespit sonuçları (kutu + sınıf + güven)

Matematiksel Formül:
-------------------
Confidence = P(Object) × IOU(pred, truth)
Class Score = P(Class_i | Object) × Confidence

YOLO Sürümleri:
---------------
• YOLOv1 (2016): İlk versiyon, 45 FPS
• YOLOv2/YOLO9000 (2017): Batch normalization, anchor boxes
• YOLOv3 (2018): Multi-scale detection, Darknet-53
• YOLOv4 (2020): CSPDarknet53, PANet, SPP
• YOLOv5 (2020): PyTorch implementasyonu, kullanım kolaylığı
• YOLOv8 (2023): Ultralytics, en güncel ve performanslı sürüm

Bu uygulamada YOLOv8 kullanılmaktadır.


COCO Dataset Sınıfları (80 sınıf):
----------------------------------
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush

Geliştirici: BENGÜSU DUMAN
"""

# ==================== KÜTÜPHANE İMPORTLARI ====================

# os: Dosya ve dizin işlemleri için
import os
# sys: Sistem işlemleri ve hata yakalama için
import sys
# tkinter: GUI (Grafiksel Kullanıcı Arayüzü) oluşturmak için
from tkinter import *
from tkinter import filedialog, messagebox

# cv2 (OpenCV): Görüntü işleme kütüphanesi
import cv2
# numpy: Sayısal hesaplamalar ve dizi işlemleri için
import numpy as np
# PIL: Python Imaging Library - Tkinter ile görüntü göstermek için
from PIL import Image, ImageTk

# ultralytics: YOLOv8 modeli için
# Kurulum: pip install ultralytics
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("UYARI: ultralytics kütüphanesi yüklü değil!")
    print("Kurulum için: pip install ultralytics")

# ==================== GLOBAL DEĞİŞKENLER ====================

# original_image: Yüklenen orijinal görüntüyü saklar (OpenCV formatında)
original_image = None

# detected_objects: Tespit edilen nesnelerin listesi
detected_objects = []

# yolo_model: YOLO modeli (lazy loading - ihtiyaç olduğunda yüklenir)
yolo_model = None


# ==================== YARDIMCI FONKSİYONLAR ====================

def create_placeholder(width=300, height=300):
    """
    Boş placeholder görüntüsü oluşturur.

    Bu fonksiyon, henüz görüntü yüklenmediğinde gösterilecek
    mavi bir placeholder görüntüsü oluşturur.

    Parametreler:
        width (int): Görüntü genişliği (piksel)
        height (int): Görüntü yüksekliği (piksel)

    Dönüş:
        ImageTk.PhotoImage: Tkinter uyumlu görüntü nesnesi
    """
    # Mavi renkli boş bir görüntü oluştur
    img = Image.new('RGB', (width, height), color='#4A90D9')
    return ImageTk.PhotoImage(img)


def load_yolo_model():
    """
    YOLOv8 modelini yükler (lazy loading).

    Model ilk kullanımda yüklenir ve sonraki kullanımlar için
    önbellekte tutulur. Bu, başlangıç süresini hızlandırır.

    YOLOv8 Model Boyutları:
    - yolov8n.pt: Nano (en hızlı, en az doğru)
    - yolov8s.pt: Small
    - yolov8m.pt: Medium
    - yolov8l.pt: Large
    - yolov8x.pt: Extra Large (en yavaş, en doğru)

    Dönüş:
        YOLO: Yüklenmiş YOLO modeli veya None (hata durumunda)
    """
    global yolo_model

    # Model zaten yüklüyse tekrar yükleme
    if yolo_model is not None:
        return yolo_model

    # YOLO kütüphanesi yüklü değilse
    if not YOLO_AVAILABLE:
        messagebox.showerror(
            "Hata",
            "YOLO kütüphanesi yüklü değil!\n\n"
            "Kurulum için terminalde çalıştırın:\n"
            "pip install ultralytics"
        )
        return None

    try:
        # Bilgi mesajı göster
        info_text.set("YOLO modeli yükleniyor... Lütfen bekleyin.")
        root.update()

        # YOLOv8 nano modelini yükle (hız için)
        # İlk çalıştırmada model otomatik indirilir (~6MB)
        yolo_model = YOLO('yolov8n.pt')

        info_text.set("YOLO modeli yüklendi!")
        return yolo_model

    except Exception as e:
        messagebox.showerror("Hata", f"Model yüklenemedi:\n{str(e)}")
        return None


# ==================== GÖRÜNTÜ YÜKLEME FONKSİYONU ====================

def load_image():
    """
    Kullanıcıdan görüntü dosyası seçmesini ister ve yükler.

    İşlem Adımları:
    ---------------
    1. Dosya seçme dialogu açılır (jpg, png, bmp desteklenir)
    2. Seçilen dosya binary olarak okunur
    3. OpenCV ile görüntü decode edilir (BGR formatında)
    4. Görüntü 300x300 piksel olarak yeniden boyutlandırılır
    5. BGR'den RGB'ye dönüştürülür (Tkinter için)
    6. Tkinter Label'da gösterilir
    7. Önceki tespit sonuçları temizlenir
    """
    global original_image, detected_objects

    # Dosya seçme dialogu aç
    # filetypes: Gösterilecek dosya türlerini filtreler
    file_path = filedialog.askopenfilename(
        title="Görüntü Seçin",
        filetypes=[
            ("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Tüm Dosyalar", "*.*")
        ]
    )

    # Kullanıcı iptal ettiyse çık
    if not file_path:
        return

    try:
        # Dosyayı binary modda oku
        # Bu yöntem Unicode karakterli dosya yollarını destekler
        with open(file_path, 'rb') as f:
            # Dosya içeriğini numpy array'e çevir
            file_bytes = np.frombuffer(f.read(), np.uint8)
            # OpenCV ile görüntüyü decode et (BGR formatında)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Görüntü okunamadıysa hata ver
        if img is None:
            messagebox.showerror(
                "Hata",
                "Görüntü yüklenemedi!\n"
                "Lütfen geçerli bir görüntü dosyası seçin."
            )
            return

        # Orijinal boyutları kaydet
        original_h, original_w = img.shape[:2]

        # Görüntüyü 300x300 piksel olarak yeniden boyutlandır
        # INTER_AREA: Küçültme için en iyi interpolasyon yöntemi
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

        # Global değişkene kaydet (YOLO için BGR formatında tut)
        original_image = img

        # BGR'den RGB'ye çevir (OpenCV BGR, Tkinter/PIL RGB kullanır)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # PIL Image'a çevir
        img_pil = Image.fromarray(img_rgb)

        # Tkinter PhotoImage'a çevir
        img_tk = ImageTk.PhotoImage(img_pil)

        # Label'ı güncelle
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Referansı tut (garbage collection önleme)

        # Bilgi metnini güncelle
        info_text.set(
            f"Yüklendi: {os.path.basename(file_path)}\n"
            f"Orijinal Boyut: {original_w}x{original_h}"
        )

        # Önceki tespit sonuçlarını temizle
        detected_objects.clear()
        update_table()

    except Exception as e:
        messagebox.showerror("Hata", f"Görüntü yüklenirken hata:\n{str(e)}")


# ==================== YOLO NESNE TESPİTİ ====================

def apply_yolo():
    """
    YOLOv8 algoritmasını kullanarak nesne tespiti yapar.

    YOLO Algoritma Adımları:
    ------------------------
    1. Görüntü modele gönderilir
    2. Özellik çıkarımı (feature extraction) yapılır
    3. Sınırlayıcı kutu (bounding box) tahminleri yapılır
    4. Sınıf olasılıkları hesaplanır
    5. Non-Maximum Suppression (NMS) uygulanır
    6. Güven eşiğini geçen tespitler döndürülür

    YOLOv8 Çıktı Formatı:
    --------------------
    Her tespit için:
    - boxes.xyxy: [x1, y1, x2, y2] koordinatları
    - boxes.conf: Güven skoru (0-1 arası)
    - boxes.cls: Sınıf indeksi (0-79 arası)
    """
    global detected_objects

    # Görüntü kontrolü
    if original_image is None:
        messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
        return

    # YOLO modelini yükle (lazy loading)
    model = load_yolo_model()
    if model is None:
        return

    try:
        # Bilgi güncelle
        info_text.set("YOLO çalışıyor... Lütfen bekleyin.")
        root.update()

        # ========== YOLO TESPİTİ ==========
        # model() fonksiyonu görüntüyü işler ve sonuçları döndürür
        # conf: Minimum güven eşiği (0.25 = %25)
        # verbose: Konsol çıktısını kapat
        results = model(original_image, conf=0.25, verbose=False)

        # Sonuç nesnesini al
        result = results[0]

        # Tespit edilen nesneleri temizle
        detected_objects.clear()

        # Görüntü kopyası üzerine çizim yap
        output_image = original_image.copy()

        # ========== TESPİTLERİ İŞLE ==========
        # Her tespit için döngü
        for box in result.boxes:
            # Sınırlayıcı kutu koordinatlarını al [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Güven skorunu al (0-1 arası)
            confidence = float(box.conf[0].cpu().numpy())

            # Sınıf indeksini al
            class_id = int(box.cls[0].cpu().numpy())

            # Sınıf adını al (COCO dataset sınıf isimleri)
            class_name = result.names[class_id]

            # Tespit bilgisini kaydet
            detected_objects.append({
                'class': class_name,  # Nesne sınıfı
                'confidence': confidence,  # Güven skoru
                'bbox': (x1, y1, x2, y2),  # Sınırlayıcı kutu
                'area': (x2 - x1) * (y2 - y1)  # Alan (piksel²)
            })

            # ========== GÖRÜNTÜ ÜZERİNE ÇİZİM ==========
            # Rastgele renk oluştur (her sınıf için tutarlı)
            np.random.seed(class_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Sınırlayıcı kutuyu çiz
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

            # Etiket metni oluştur
            label = f"{class_name}: {confidence:.2f}"

            # Etiket arka planı için boyut hesapla
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Etiket arka planını çiz (okunabilirlik için)
            cv2.rectangle(
                output_image,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1  # Dolu dikdörtgen
            )

            # Etiket metnini yaz
            cv2.putText(
                output_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font boyutu
                (255, 255, 255),  # Beyaz renk
                1  # Kalınlık
            )

        # ========== SONUCU GÖSTER ==========
        show_result_image(output_image)

        # Tabloyu güncelle
        update_table()

        # Bilgi güncelle
        total = len(detected_objects)
        unique = len(set(obj['class'] for obj in detected_objects))
        info_text.set(f"Tespit tamamlandı! {total} nesne bulundu ({unique} farklı sınıf)")

        # Algoritma açıklamasını göster
        show_algorithm_info()

    except Exception as e:
        messagebox.showerror("Hata", f"YOLO çalıştırılırken hata:\n{str(e)}")
        info_text.set("Hata oluştu!")


# ==================== SONUÇ GÖRÜNTÜSÜNÜ GÖSTERME ====================

def show_result_image(img):
    """
    İşlenmiş görüntüyü ana görüntü alanında gösterir.

    Parametreler:
        img: OpenCV formatında görüntü (BGR)
    """
    # BGR'den RGB'ye çevir
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # PIL ve Tkinter formatına çevir
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Label'ı güncelle
    image_label.config(image=img_tk)
    image_label.image = img_tk


# ==================== ALGORİTMA BİLGİSİNİ GÖSTERME ====================

def show_algorithm_info():
    """
    YOLO algoritma açıklamasını metin alanında gösterir.
    """
    info = """
╔═══════════════════════════════════════════╗
║           YOLOv8 ALGORİTMASI              ║
╚═══════════════════════════════════════════╝

AÇIKLAMA:
YOLO (You Only Look Once), görüntüyü tek bir
ileri geçişte işleyerek gerçek zamanlı nesne
tespiti yapan derin öğrenme algoritmasıdır.

ALGORİTMA ADIMLARI:
1. Görüntü S×S ızgaraya bölünür
2. Her hücre B sınırlayıcı kutu tahmin eder
3. Sınıf olasılıkları hesaplanır
4. Non-Maximum Suppression uygulanır
5. Final tespitler döndürülür

FORMÜLLER:
• Confidence = P(Obj) × IOU(pred, truth)
• Class Score = P(Class|Obj) × Confidence
• IOU = Kesişim Alanı / Birleşim Alanı

YOLOv8 ÖZELLİKLERİ:
• Anchor-free detection
• Mosaic augmentation
• CSPDarknet backbone
• PANet neck
• Decoupled head

KULLANILAN MODEL:
• YOLOv8n (Nano) - Hız optimizeli
• 80 COCO sınıfı desteklenir
• Güven eşiği: 0.25

KAYNAKLAR:
1. Redmon et al. (2016) - YOLO Paper
2. Redmon & Farhadi (2018) - YOLOv3
3. Ultralytics YOLOv8 Docs (2023)
"""
    algorithm_text.delete(1.0, END)
    algorithm_text.insert(END, info)


# ==================== SONUÇ TABLOSUNU GÜNCELLEME ====================

def update_table():
    """
    Tespit edilen nesneleri tablo formatında gösterir.

    Tablo İçeriği:
    - Nesne sınıfı
    - Güven skoru
    - Sınırlayıcı kutu alanı
    - Toplam ve benzersiz nesne sayısı
    """
    # Mevcut içeriği temizle
    table_text.delete(1.0, END)

    # Tespit yoksa bilgi mesajı göster
    if not detected_objects:
        table_text.insert(END, "╔═══════════════════════════════════╗\n")
        table_text.insert(END, "║      TESPİT SONUÇLARI             ║\n")
        table_text.insert(END, "╠═══════════════════════════════════╣\n")
        table_text.insert(END, "║ Henüz tespit yapılmadı.           ║\n")
        table_text.insert(END, "║                                   ║\n")
        table_text.insert(END, "║ 1. Görüntü Yükle butonuna tıklayın║\n")
        table_text.insert(END, "║ 2. YOLO butonuna tıklayın         ║\n")
        table_text.insert(END, "╚═══════════════════════════════════╝\n")
        return

    # ========== TABLO BAŞLIĞI ==========
    table_text.insert(END, "╔═══════════════════════════════════════╗\n")
    table_text.insert(END, "║      YOLO TESPİT SONUÇLARI            ║\n")
    table_text.insert(END, "╠═══════════════════════════════════════╣\n")

    # Sınıf bazında grupla
    class_counts = {}
    for obj in detected_objects:
        cls = obj['class']
        if cls not in class_counts:
            class_counts[cls] = {'count': 0, 'max_conf': 0}
        class_counts[cls]['count'] += 1
        class_counts[cls]['max_conf'] = max(
            class_counts[cls]['max_conf'],
            obj['confidence']
        )

    # ========== SINIF BAZLI SONUÇLAR ==========
    table_text.insert(END, "║ SINIF             ADET    GÜVEN      ║\n")
    table_text.insert(END, "╟───────────────────────────────────────╢\n")

    # Her sınıf için satır ekle
    for cls, data in sorted(class_counts.items(),
                            key=lambda x: x[1]['count'],
                            reverse=True):
        conf_percent = data['max_conf'] * 100
        table_text.insert(
            END,
            f"║ {cls:<17} {data['count']:<7} %{conf_percent:<6.1f}   ║\n"
        )

    # ========== ÖZET BİLGİLER ==========
    table_text.insert(END, "╠═══════════════════════════════════════╣\n")
    table_text.insert(END, "║            ÖZET BİLGİLER              ║\n")
    table_text.insert(END, "╟───────────────────────────────────────╢\n")

    total = len(detected_objects)
    unique = len(class_counts)
    avg_conf = np.mean([obj['confidence'] for obj in detected_objects]) * 100

    table_text.insert(END, f"║ Toplam Tespit: {total:<23} ║\n")
    table_text.insert(END, f"║ Farklı Sınıf:  {unique:<23} ║\n")
    table_text.insert(END, f"║ Ort. Güven:    %{avg_conf:<21.1f} ║\n")

    # ========== 5 FARKLI NESNE KONTROLÜ ==========
    table_text.insert(END, "╠═══════════════════════════════════════╣\n")
    if unique >= 5:
        table_text.insert(END, "║ ✓ 5+ farklı nesne tespit edildi!     ║\n")
    else:
        table_text.insert(END, f"║ ⚠ {unique}/5 farklı nesne tespit edildi  ║\n")

    table_text.insert(END, "╚═══════════════════════════════════════╝\n")


# ==================== ANA PENCERE OLUŞTURMA ====================

# Ana Tkinter penceresi oluştur
root = Tk()
root.title("YOLO Nesne Tespit - Ödev 5")
root.geometry("800x700")
root.configure(bg='#E8E0D5')  # Açık bej arka plan (örnek formla uyumlu)

# ==================== ÜST BÖLÜM (Butonlar ve Görüntü) ====================

# Üst çerçeve
top_frame = Frame(root, bg='#E8E0D5', bd=3, relief=GROOVE)
top_frame.pack(pady=20, padx=20, fill=X)

# Sol taraf - Butonlar
button_frame = Frame(top_frame, bg='#E8E0D5')
button_frame.pack(side=LEFT, padx=30, pady=20)

# Görüntü Yükle butonu
load_btn = Button(
    button_frame,
    text="Görüntü Yükle",
    command=load_image,
    width=20,
    height=2,
    font=("Arial", 12, "bold"),
    bg='#1A1A1A',  # Siyah arka plan
    fg='white',  # Beyaz yazı
    activebackground='#333333',
    activeforeground='white',
    relief=RAISED,
    bd=3,
    cursor='hand2'  # Fare imleci
)
load_btn.pack(pady=15)

# YOLO butonu
yolo_btn = Button(
    button_frame,
    text="YOLO",
    command=apply_yolo,
    width=20,
    height=2,
    font=("Arial", 12, "bold"),
    bg='#1A1A1A',
    fg='white',
    activebackground='#333333',
    activeforeground='white',
    relief=RAISED,
    bd=3,
    cursor='hand2'
)
yolo_btn.pack(pady=15)

# Sağ taraf - Görüntü alanı
image_frame = Frame(top_frame, bg='#4A90D9', bd=3, relief=SOLID)
image_frame.pack(side=RIGHT, padx=30, pady=20)

# Görüntü başlığı
image_title = Label(
    image_frame,
    text="Görüntü Örnek\n(300x300 çözünürlük)",
    font=("Arial", 11),
    bg='#4A90D9',
    fg='white'
)
image_title.pack(pady=5)

# Placeholder görüntüsü
placeholder_img = create_placeholder(300, 300)

# Görüntü label'ı
image_label = Label(
    image_frame,
    image=placeholder_img,
    bg='#4A90D9'
)
image_label.image = placeholder_img
image_label.pack(padx=10, pady=10)

# ==================== BİLGİ ETİKETİ ====================

info_text = StringVar()
info_text.set("Görüntü yükleyip YOLO butonuna tıklayın")

info_label = Label(
    root,
    textvariable=info_text,
    font=("Arial", 10),
    bg='#E8E0D5',
    fg='#333333'
)
info_label.pack(pady=5)

# ==================== ALT BÖLÜM (Açıklama ve Tablo) ====================

bottom_frame = Frame(root, bg='#E8E0D5')
bottom_frame.pack(pady=10, padx=20, fill=BOTH, expand=True)

# Sol alt - Algoritma açıklaması
left_bottom = Frame(bottom_frame, bg='#E8E0D5')
left_bottom.pack(side=LEFT, padx=10, fill=BOTH, expand=True)

algo_title = Label(
    left_bottom,
    text="ALGORİTMA AÇIKLAMASI",
    font=("Arial", 11, "bold"),
    bg='#E8E0D5',
    fg='#333333'
)
algo_title.pack(pady=5)

# Algoritma metin alanı
algorithm_text = Text(
    left_bottom,
    width=40,
    height=16,
    font=("Consolas", 9),
    bg='#FFFFFF',
    fg='#333333',
    relief=SUNKEN,
    bd=2
)
algorithm_text.pack(fill=BOTH, expand=True)
algorithm_text.insert(END, "YOLO butonuna tıkladığınızda\nalgorıtma açıklaması burada görünecek.\n\n")
algorithm_text.insert(END, "YOLO (You Only Look Once):\n")
algorithm_text.insert(END, "Gerçek zamanlı nesne tespiti için\ngeliştirilmiş derin öğrenme algoritması.")

# Sağ alt - Sonuç tablosu
right_bottom = Frame(bottom_frame, bg='#E8E0D5')
right_bottom.pack(side=RIGHT, padx=10, fill=BOTH, expand=True)

table_title = Label(
    right_bottom,
    text="SONUÇ TABLOSU",
    font=("Arial", 11, "bold"),
    bg='#E8E0D5',
    fg='#333333'
)
table_title.pack(pady=5)

# Tablo metin alanı
table_text = Text(
    right_bottom,
    width=45,
    height=16,
    font=("Consolas", 9),
    bg='#FFFFFF',
    fg='#333333',
    relief=SUNKEN,
    bd=2
)
table_text.pack(fill=BOTH, expand=True)

# Başlangıç tablo içeriği
update_table()

# ==================== ÖĞRENCİ BİLGİSİ ====================

student_frame = Frame(root, bg='#E8E0D5')
student_frame.pack(side=BOTTOM, pady=15)

student_label = Label(
    student_frame,
    text="  BENGÜSU DUMAN  ",
    font=("Arial", 11),
    bg='white',
    fg='black',
    bd=1,
    relief=SOLID,
    padx=20,
    pady=5
)
student_label.pack()

# ==================== UYGULAMAYI BAŞLAT ====================

# Tkinter ana döngüsünü başlat
# Bu döngü, pencereyi açık tutar ve kullanıcı etkileşimlerini işler
root.mainloop()

