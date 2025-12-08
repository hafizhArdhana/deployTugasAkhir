# 🧴 Acne Detection & Treatment Recommendation System

Sistem deteksi jerawat berbasis AI menggunakan YOLO dengan rekomendasi perawatan otomatis menggunakan rule-based system.

## 👤 Identitas

**Nama:** Hafizh Putra Ardhana  
**NIM:** 1305220049  
**Program Studi:** S1 Sains Data  
**Fakultas:** Informatika  

**Dosen Pembimbing:**
- Pembimbing 1: Dra. INDWIARTI, M.Si
- Pembimbing 2: Dr. PUTU HARRY G., S.Si., M.Si., M.Sc

---

## 📋 Deskripsi Sistem

Aplikasi web berbasis Streamlit yang mengintegrasikan model YOLO untuk deteksi jerawat dengan sistem rekomendasi perawatan berbasis aturan (rule-based). Sistem melakukan preprocessing otomatis pada gambar input dan memberikan rekomendasi treatment berdasarkan jenis lesi yang terdeteksi.

---

## 🛠️ Teknologi & Library

- **ultralytics**: Framework YOLO untuk object detection
- **streamlit**: Framework web application
- **pillow (PIL)**: Image processing
- **matplotlib**: Visualisasi hasil
- **pandas**: Data manipulation
- **numpy**: Array operations

---

## 📁 Struktur Code

### **1. Model Loading (`load_model()`)**
```python
model_path = os.getenv("MODEL_PATH", "best.pt")

@st.cache_resource
def load_model():
    model = YOLO(model_path)
    return model
```
**Fungsi:**
- Load model YOLO dari file `best.pt`
- Menggunakan `@st.cache_resource` untuk caching agar model hanya di-load sekali
- Path model dinamis menggunakan environment variable atau default ke `best.pt`

---

### **2. Preprocessing (`preprocess_image()`)**
```python
def preprocess_image(img: Image.Image, target_size: Tuple[int, int] = (640, 640))
```
**Fungsi:**
- Resize gambar ke 640x640 piksel menggunakan LANCZOS resampling
- Normalisasi nilai piksel ke rentang 0-1 (divide by 255)
- Return: gambar yang sudah di-resize dan array yang sudah dinormalisasi

**Alasan:**
- YOLO model dilatih dengan input size 640x640
- Normalisasi memastikan konsistensi input untuk model

---

### **3. Rule-Based Recommendation (`treatment_map`)**
```python
treatment_map = {
    "blackheads": {...},
    "whiteheads": {...},
    "papules": {...},
    "pustules": {...},
    "nodules": {...},
    "dark_spot": {...}
}
```
**Fungsi:**
- Dictionary berisi mapping antara jenis lesi dengan treatment yang direkomendasikan
- Setiap jenis lesi memiliki list treatment dan catatan medis
- Mencakup treatment konvensional dan nanoteknologi

**Kegunaan:**
- Memberikan rekomendasi perawatan otomatis berdasarkan hasil deteksi
- Tidak perlu model ML tambahan untuk rekomendasi

---

### **4. Inference (`run_inference_on_image()`)**
```python
def run_inference_on_image(img: Image.Image, conf: float = 0.25, iou: float = 0.45)
```
**Fungsi:**
- Melakukan preprocessing pada gambar input
- Menjalankan YOLO model prediction dengan parameter confidence dan IoU threshold
- Mengembalikan gambar dengan bounding boxes dan list deteksi

**Parameter:**
- `conf`: Confidence threshold (0.1-0.9) - minimum confidence untuk deteksi
- `iou`: IoU threshold (0.1-0.9) - untuk Non-Maximum Suppression (NMS)

**Output:**
- Annotated image dengan bounding boxes
- List deteksi berisi: bbox coordinates, confidence, class_id, class_name

---

### **5. Recommendation System (`get_recommendation()` & `summarize_recommendations()`)**
```python
def get_recommendation(class_name: str)
def summarize_recommendations(detections: List[Dict[str, Any]])
```
**Fungsi:**
- `get_recommendation()`: Mengambil treatment dari `treatment_map` berdasarkan class name
- `summarize_recommendations()`: Agregasi rekomendasi dari semua deteksi dan menghilangkan duplikasi

**Kegunaan:**
- Memberikan rekomendasi per jenis lesi
- Memberikan ringkasan treatment tanpa duplikasi untuk efisiensi

---

### **6. Data Export (`detections_to_df()` & `image_to_bytes()`)**
```python
def detections_to_df(detections: List[Dict[str, Any]]) -> pd.DataFrame
def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes
```
**Fungsi:**
- `detections_to_df()`: Konversi list deteksi ke pandas DataFrame untuk visualisasi tabel
- `image_to_bytes()`: Konversi PIL Image ke bytes untuk download

**Kegunaan:**
- Export hasil deteksi dalam format JSON dan PNG
- Memudahkan analisis lebih lanjut

---

### **7. Streamlit UI (`streamlit_app()`)**
```python
def streamlit_app()
```
**Fungsi:**
- Main application interface dengan 3 tabs: Upload, Hasil Deteksi, Tentang
- Sidebar untuk konfigurasi confidence dan IoU threshold
- Upload gambar atau ambil foto dari kamera
- Preview preprocessing (before/after)
- Visualisasi hasil deteksi dengan metrics
- Download hasil (PNG & JSON)

**Fitur UI:**
- Real-time progress indicator saat preprocessing
- Before/after comparison gambar
- Statistik deteksi (total, jenis, avg confidence)
- Rekomendasi treatment per jenis lesi
- Ringkasan treatment tanpa duplikasi

---

## 🔧 Spesifikasi Model

- **Architecture**: YOLO (You Only Look Once)
- **Input Size**: 640x640 pixels
- **Model Size**: 22 MB
- **Classes**: 6 jenis lesi
  - blackheads (komedo hitam)
  - whiteheads (komedo putih)
  - papules (papul)
  - pustules (pustul)
  - nodules (nodul)
  - dark_spot (bekas jerawat)

---

## 🔄 Workflow Sistem

1. **Input**: User upload gambar atau ambil foto
2. **Preprocessing**: 
   - Resize ke 640x640px
   - Normalisasi piksel ke 0-1
3. **Detection**: YOLO model inference
4. **Post-processing**: 
   - Filter berdasarkan confidence threshold
   - Non-Maximum Suppression (NMS) dengan IoU threshold
5. **Recommendation**: 
   - Mapping deteksi ke treatment_map
   - Agregasi dan deduplikasi rekomendasi
6. **Output**: 
   - Gambar dengan bounding boxes
   - Statistik deteksi
   - Rekomendasi treatment
   - Export JSON & PNG

---

## 📊 Treatment Categories

| Jenis Lesi | Treatment Utama |
|------------|-----------------|
| **Blackheads** | Retinoid topikal, Asam salisilat |
| **Whiteheads** | Retinoid topikal, BHA |
| **Papules** | Benzoyl peroxide, Antibiotik topikal |
| **Pustules** | Antibiotik oral/topikal, Asam azelaat |
| **Nodules** | Isotretinoin oral, Kortikosteroid intralesi |
| **Dark Spots** | Niacinamide, Vitamin C, Sunscreen |

Semua treatment mendukung penghantaran nanoteknologi (nanoemulsi, liposom, nanopartikel).

---

## 📝 Catatan Teknis

### **Preprocessing**
- Menggunakan LANCZOS resampling untuk kualitas resize terbaik
- Normalisasi 0-1 sesuai dengan training model YOLO

### **Caching**
- `@st.cache_resource` untuk model loading (persistent across reruns)
- Menghindari reload model setiap kali user interact

### **Dynamic Path**
- Model path menggunakan environment variable untuk fleksibilitas deployment
- Default ke `best.pt` di folder yang sama dengan script

### **Error Handling**
- Validasi keberadaan model file
- Informative error messages untuk debugging

---

## 📄 Lisensi

Tugas Akhir - S1 Sains Data, Fakultas Informatika

---

## 🙏 Acknowledgments

- Dra. INDWIARTI, M.Si (Pembimbing 1)
- Dr. PUTU HARRY G., S.Si., M.Si., M.Sc (Pembimbing 2)
- Ultralytics YOLO Team
- Streamlit Community
