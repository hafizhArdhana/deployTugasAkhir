from ultralytics import YOLO
import os, json, io
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import time
import streamlit as st

# ====== Path model terbaik hasil trainingmu ======
# Dinamis: gunakan environment variable atau default ke best.pt di folder yang sama
model_path = os.getenv("MODEL_PATH", "best.pt")

# Load model hanya sekali dengan caching
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.info("💡 Ensure best.pt file is in the same folder or set MODEL_PATH environment variable")
        st.stop()
    model = YOLO(model_path)
    print("✅ Model loaded:", model_path)
    return model

# ====== Rule-based Recommendation Dictionary ======
treatment_map = {
    # =======================
    # 1. KOMEDO (Blackhead / Whitehead)
    # =======================
    "blackheads": {
        "treatments": [
            "Topical Retinoids (adapalene/tretinoin)",
            "Salicylic Acid",
            "Benzoyl Peroxide",
            "Azelaic Acid",
            "Nanotechnology delivery: nanoemulsion, nanoparticle, solid lipid nanoparticle"
        ],
        "note": (
            "Topical retinoids are effective for open/closed comedones. "
            "Salicylic acid for pore exfoliation. "
            "Nanotechnology delivery (nanoemulsions, liposomes, solid lipid nanoparticles) "
            "improves active ingredient stability and can enhance therapy effectiveness "
            "with minimal irritation."
        )
    },

    "whiteheads": {
        "treatments": [
            "Topical Retinoids (adapalene/tretinoin)",
            "Salicylic Acid",
            "Benzoyl Peroxide",
            "Azelaic Acid",
            "Nanotechnology delivery: nanoemulsion, nanoparticle, solid lipid nanoparticle"
        ],
        "note": (
            "Topical retinoids are effective for closed comedones. "
            "Exfoliation with BHA helps unclog pores. "
            "Nanotechnology delivery can improve active ingredient penetration."
        )
    },

    # =======================
    # 2. PAPUL
    # =======================
    "papules": {
        "treatments": [
            "Topical Retinoids",
            "Benzoyl Peroxide",
            "Topical Antibiotics (clindamycin / erythromycin)",
            "Retinoid + Benzoyl Peroxide combination",
            "Azelaic Acid",
            "Nanotechnology delivery: nanoemulsion, nanoparticle, solid lipid nanoparticle"
        ],
        "note": (
            "Combination of retinoid and benzoyl peroxide to reduce inflammation. "
            "Topical antibiotics (clindamycin, erythromycin) are effective for acne-causing bacteria. "
            "Azelaic acid works as an anti-inflammatory and anti-comedogenic agent. "
            "Use of nanotechnology helps improve local effectiveness."
        )
    },

    # =======================
    # 3. PUSTUL
    # =======================
    "pustules": {
        "treatments": [
            "Topical Retinoids",
            "Benzoyl Peroxide",
            "Topical/Oral Antibiotics",
            "Azelaic Acid",
            "Nanotechnology delivery: nanoemulsion, nanoparticle, solid lipid nanoparticle"
        ],
        "note": (
            "For ample pustules, consider oral antibiotics (doxycycline/minocycline). "
            "Azelaic acid is beneficial for reducing inflammation and hyperpigmentation. "
            "Nanotechnology can improve active ingredient penetration."
        )
    },

    # =======================
    # 4. NODUL
    # =======================
    "nodules": {
        "treatments": [
            "Oral Isotretinoin",
            "Oral Antibiotics",
            "Intralesional Corticosteroids",
            "Topical Retinoids",
            "Nanotechnology delivery: liposomes, nanoparticles, polymeric nanoparticles"
        ],
        "note": (
            "Isotretinoin is the main therapy for severe cases. "
            "Oral antibiotics help reduce inflammation. "
            "Intralesional corticosteroids are used for large nodules. "
            "Nanoparticles help improve active ingredient stability in severe lesions."
        )
    },

    # =======================
    # 5. BEKAS JERAWAT / PIH / PIE
    # =======================
    "dark_spot": {
        "treatments": [
            "Niacinamide",
            "Vitamin C",
            "Topical Retinoids",
            "Azelaic Acid",
            "Sunscreen",
            "Nanotechnology delivery: liposomes, nanoemulsion, silver/gold/silver nanoparticles"
        ],
        "note": (
            "Focus on prevention and hyperpigmentation repair. "
            "Niacinamide and vitamin C help brighten. "
            "Sunscreen is crucial to prevent worsening PIH. "
            "Nanotechnology improves stability and penetration of active ingredients."
        )
    }
}


# ====== Fungsi Preprocessing ======
def preprocess_image(img: Image.Image, target_size: Tuple[int, int] = (640, 640)) -> Tuple[Image.Image, np.ndarray]:
    """
    Preprocessing gambar: resize ke 640x640 dan normalisasi piksel ke 0-1
    Returns: (processed_pil_image, normalized_array)
    """
    # Resize gambar ke target size
    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Konversi ke numpy array dan normalisasi ke 0-1
    img_array = np.array(resized_img).astype(np.float32) / 255.0
    
    return resized_img, img_array

# ====== Fungsi utilitas ======
def get_recommendation(class_name: str):
    cls = class_name.lower().strip().replace("-", "_").replace(" ", "_")
    if cls in treatment_map:
        return treatment_map[cls]
    else:
        return {
            "treatments": ["No recommendation available for this class."],
            "note": "Acne type not defined in system rules."
        }

def detect_and_recommend(image_path, conf=0.25):
    """Jalankan deteksi pada satu gambar dan tampilkan hasil serta rekomendasi."""
    print(f"\n📸 Processing image: {image_path}")
    
    # Load dan preprocess gambar
    img = Image.open(image_path).convert("RGB")
    processed_img, normalized_array = preprocess_image(img)
    
    model = load_model()  # Gunakan cached model
    results = model.predict(source=processed_img, conf=conf, save=True, verbose=False)
    res = results[0]

    if len(res.boxes) == 0:
        print("❌ No acne detected in this image.")
        return

    # Ambil hasil deteksi
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    names = res.names

    all_recos = []

    print("\n🔍 Detection Results & Recommendations:")
    for xy, cf, c in zip(boxes, confs, clss):
        name = names[int(c)]
        reco = get_recommendation(name)
        print(f"\n🧩 Lesion Type: {name} ({cf:.2f})")
        print("💊 Chemical Treatment Recommendations:")
        for t in reco["treatments"]:
            print(f"   - {t}")
        print(f"📋 Note: {reco['note']}")
        all_recos.extend(reco["treatments"])

    # Ringkasan total
    summary = sorted(set(all_recos))
    print("\n📦 Recommendations Summary (without duplicates):")
    for t in summary:
        print(" -", t)

    # Tampilkan gambar hasil deteksi
    plt.imshow(Image.open(Path(res.save_dir) / os.path.basename(image_path)))
    plt.axis("off")
    plt.show()


# ====== Util untuk Streamlit UI (dengan preprocessing) ======
def run_inference_on_image(img: Image.Image, conf: float = 0.25, iou: float = 0.45, progress_callback=None) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Prediksi dengan preprocessing: resize 640x640 + normalisasi 0-1."""
    
    model = load_model()  # Gunakan cached model
    
    # Step 1: Resize
    if progress_callback:
        progress_callback("🔄 Resizing image to 640x640...", 0.2)
    time.sleep(0.3)  # Simulasi proses
    processed_img, normalized_array = preprocess_image(img, (640, 640))
    
    # Step 2: Normalisasi
    if progress_callback:
        progress_callback("⚡ Normalizing pixels to 0-1 range...", 0.5)
    time.sleep(0.3)  # Simulasi proses
    
    # Step 3: Inference
    if progress_callback:
        progress_callback("🧠 Running AI detection...", 0.8)
    
    results = model.predict(source=processed_img, conf=conf, iou=iou, save=False, verbose=False)
    res = results[0]

    # Gambar anotasi (np.array BGR) -> PIL RGB
    plotted = res.plot()
    annotated = Image.fromarray(plotted[:, :, ::-1])

    detections: List[Dict[str, Any]] = []
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names
        for xy, cf, c in zip(boxes, confs, clss):
            detections.append({
                "bbox": [float(x) for x in xy.tolist()],
                "conf": float(cf),
                "class_id": int(c),
                "class_name": names[int(c)],
            })
    
    if progress_callback:
        progress_callback("✅ Done!", 1.0)
    
    return annotated, detections


def summarize_recommendations(detections: List[Dict[str, Any]]):
    recos_by_class: Dict[str, Dict[str, Any]] = {}
    all_treatments: List[str] = []
    for d in detections:
        name = d["class_name"]
        reco = get_recommendation(name)
        recos_by_class.setdefault(name, reco)
        all_treatments.extend(reco["treatments"])
    return recos_by_class, sorted(set(all_treatments))


def detections_to_df(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        rows.append({
            "class": d["class_name"],
            "confidence": round(d["conf"], 3),
            "x1": round(x1, 2), "y1": round(y1, 2),
            "x2": round(x2, 2), "y2": round(y2, 2),
        })
    return pd.DataFrame(rows)


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.read()


def streamlit_app():
    import streamlit as st  # import lokal agar aman saat modul di-import dari tempat lain

    st.set_page_config(page_title="Acne Detection & Recommendation", page_icon="🧴", layout="wide")

    # Enhanced styling dengan preprocessing indicators
    st.markdown(
        """
        <style>
        .stApp {background: radial-gradient(1200px 600px at 10% 10%, #0d1b2a 0%, #0b1320 35%, #0e1117 100%) !important;}
        .title {font-size: 2.5rem; font-weight: 900; letter-spacing: .3px; background: linear-gradient(45deg, #3fb950, #5b8def, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.5rem;}
        .subtitle {opacity:.85; text-align: center; font-size: 1.1rem; margin-bottom: 2rem;}
        .tag {display:inline-block; padding:8px 16px; margin:4px 6px; border-radius:999px; background:linear-gradient(45deg, #5b8def, #3fb950); color:white; font-size:0.9rem; font-weight:600;}
        .note {background:rgba(255,255,255,0.05); border:1px solid #30363d; padding:16px; border-radius:16px; backdrop-filter: blur(10px);}
        .pill {display:inline-flex; align-items:center; gap:10px; padding:10px 16px; border-radius:999px; background:linear-gradient(135deg, #161b22, #21262d); border:1px solid #30363d; margin:8px 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);}
        .pill .dot{width:12px; height:12px; border-radius:50%; background:linear-gradient(45deg, #3fb950, #58a6ff); display:inline-block; animation: pulse 2s infinite;}
        .metric-card {background:linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); border:1px solid #30363d; padding:20px; border-radius:16px; text-align:center; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.3);}
        .preprocess-info {background: linear-gradient(135deg, rgba(59, 137, 240, 0.15), rgba(63, 185, 80, 0.15)); border: 2px solid rgba(59, 137, 240, 0.3); padding: 24px; border-radius: 20px; margin: 20px 0; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(59, 137, 240, 0.2);}
        .step-indicator {display: inline-flex; align-items: center; gap: 12px; padding: 12px 20px; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); border-radius: 25px; margin: 8px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(5px);}
        .image-container {background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); border: 2px solid #30363d; border-radius: 20px; padding: 20px; margin: 10px 0; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.3);}
        .image-title {font-size: 1.2rem; font-weight: 700; margin-bottom: 15px; text-align: center; color: #58a6ff;}
        .comparison-container {display: flex; gap: 20px; align-items: stretch;}
        .before-after {flex: 1; background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border: 1px solid #30363d; border-radius: 16px; padding: 16px; backdrop-filter: blur(5px);}
        .arrow-separator {display: flex; align-items: center; justify-content: center; font-size: 2rem; color: #58a6ff; margin: 0 10px;}
        @keyframes pulse {0% {opacity: 1;} 50% {opacity: 0.5;} 100% {opacity: 1;}}
        .upload-area {background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); border: 2px dashed #30363d; border-radius: 20px; padding: 30px; text-align: center; margin: 20px 0; transition: all 0.3s ease;}
        .upload-area:hover {border-color: #58a6ff; background: linear-gradient(135deg, rgba(88, 166, 255, 0.1), rgba(88, 166, 255, 0.05));}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='title'>✨ Acne Detector</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload face/skin photo ➡ preprocessing pipeline ➡ get treatment recommendations</div>", unsafe_allow_html=True)

    # Enhanced preprocessing info box
    st.markdown("""
    <div class='preprocess-info'>
        <h3 style='text-align: center; margin-bottom: 20px; color: #58a6ff;'>🔧 Preprocessing Pipeline</h3>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
            <div class='step-indicator'>🔄 Resize → 640x640px</div>
            <div class='step-indicator'>⚡ Normalization → 0-1 range</div>
            <div class='step-indicator'>🧠 YOLO Detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # State
    if "image" not in st.session_state:
        st.session_state.image = None
    if "annotated" not in st.session_state:
        st.session_state.annotated = None
    if "detections" not in st.session_state:
        st.session_state.detections = []

    with st.sidebar:
        st.header("⚙️ Settings")
        conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05, help="Higher value means detection is more confident but might miss small objects.")
        iou = st.slider("IoU threshold", 0.1, 0.9, 0.45, 0.05, help="Controls NMS, lower value = more boxes survive.")
        
        st.divider()
        st.markdown("**🔧 Preprocessing Settings:**")
        st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid #30363d;'>
            <div style='margin-bottom: 8px;'>📐 Target size: <strong>640x640 px</strong></div>
            <div style='margin-bottom: 8px;'>📊 Normalization: <strong>0-1 range</strong></div>
            <div>🔄 Resampling: <strong>LANCZOS</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown("**🤖 Model Info:**")
        st.code(model_path, language="text")
        if os.path.exists(model_path):
            st.success("✅ Model ready")
        else:
            st.error("❌ Model not found")
        
        if st.button("🗑️ Clear Results", type="secondary", use_container_width=True):
            st.session_state.image = None
            st.session_state.annotated = None
            st.session_state.detections = []
            st.rerun()

    tab_upload, tab_results, tab_about = st.tabs(["📤 Upload", "🖼️ Detection Results", "ℹ️ About"]) 

    with tab_upload:
        # Upload section dengan styling yang lebih bagus
        st.markdown("""
        <div class='upload-area'>
            <h3 style='color: #58a6ff; margin-bottom: 15px;'>📤 Upload Image</h3>
            <p style='opacity: 0.8; margin-bottom: 20px;'>Choose image from computer or take a photo</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_up1, col_up2 = st.columns([3, 2])
        with col_up1:
            uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            cam = st.camera_input("Take photo", label_visibility="collapsed")
        with col_up2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); 
                        border: 1px solid #30363d; border-radius: 15px; padding: 20px;'>
                <h4 style='color: #3fb950; margin-bottom: 15px;'>💡 Best Photo Tips</h4>
                <div style='margin-bottom: 8px;'>☀️ Consistent and sufficient lighting</div>
                <div style='margin-bottom: 8px;'>🎯 Avoid blur/shake</div>
                <div style='margin-bottom: 8px;'>🔍 Focus on face skin area</div>
                <div>📏 Optimal distance 5-10cm</div>
            </div>
            """, unsafe_allow_html=True)

        img = None
        if cam is not None:
            img = Image.open(cam).convert("RGB")
        elif uploaded is not None:
            img = Image.open(uploaded).convert("RGB")

        if img is not None:
            st.session_state.image = img
            
            # Enhanced before/after comparison
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='image-container'>
                <h3 style='text-align: center; color: #58a6ff; margin-bottom: 25px;'>🔄 Preview Preprocessing</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col_before, col_arrow, col_after = st.columns([5, 1, 5])
            
            with col_before:
                st.markdown("""
                <div class='before-after'>
                    <div class='image-title'>📷 Original Image</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown(f"<p style='text-align: center; opacity: 0.7; margin-top: 10px;'>Resolution: {img.size[0]}x{img.size[1]} px</p>", unsafe_allow_html=True)
            
            with col_arrow:
                st.markdown("<div class='arrow-separator'>➡️</div>", unsafe_allow_html=True)
            
            with col_after:
                preview_processed, _ = preprocess_image(img, (640, 640))
                st.markdown("""
                <div class='before-after'>
                    <div class='image-title'>⚡ After Preprocessing</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(preview_processed, use_container_width=True)
                st.markdown("<p style='text-align: center; opacity: 0.7; margin-top: 10px;'>Resolution: 640x640 px | Normalization: 0-1</p>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                run = st.button("🚀 Start AI Detection", type="primary", use_container_width=True)
            
            if run:
                # Enhanced progress section
                st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(59, 137, 240, 0.1), rgba(63, 185, 80, 0.1)); 
                            border: 2px solid rgba(59, 137, 240, 0.3); border-radius: 15px; padding: 20px; margin: 20px 0;'>
                    <h4 style='text-align: center; color: #58a6ff; margin-bottom: 15px;'>🔄 Processing...</h4>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(message, progress):
                    progress_bar.progress(progress)
                    status_text.markdown(f"<div style='text-align: center; font-size: 1.1rem; margin: 10px 0;'>{message}</div>", unsafe_allow_html=True)
                
                annotated, detections = run_inference_on_image(img, conf=conf, iou=iou, progress_callback=update_progress)
                
                st.session_state.annotated = annotated
                st.session_state.detections = detections
                
                progress_bar.progress(1.0)
                status_text.success(f"✅ Preprocessing & detection complete! {len(detections)} objects found.")
                time.sleep(1)
                st.rerun()

    with tab_results:
        if st.session_state.annotated is None:
            st.markdown("""
            <div style='text-align: center; padding: 60px 20px; background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); 
                        border: 2px dashed #30363d; border-radius: 20px; margin: 20px 0;'>
                <h3 style='color: #58a6ff; margin-bottom: 15px;'>📊 No Results Yet</h3>
                <p style='opacity: 0.8;'>Please upload an image and run detection in the Upload tab first</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            ann = st.session_state.annotated
            dets = st.session_state.detections

            # Enhanced metrics cards
            st.markdown("<h3 style='text-align: center; color: #58a6ff; margin-bottom: 25px;'>📊 Detection Statistics</h3>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 2.5rem; margin-bottom: 8px;'>🎯</div>
                    <div style='font-size: 2rem; font-weight: bold; color: #3fb950;'>{len(dets)}</div>
                    <div style='opacity: 0.8;'>Total Detections</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                unique_classes = len({d['class_name'] for d in dets})
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 2.5rem; margin-bottom: 8px;'>🧩</div>
                    <div style='font-size: 2rem; font-weight: bold; color: #58a6ff;'>{unique_classes}</div>
                    <div style='opacity: 0.8;'>Unique Types</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                avg_conf = round(sum(d['conf'] for d in dets)/len(dets), 3) if dets else 0.0
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 2.5rem; margin-bottom: 8px;'>⚡</div>
                    <div style='font-size: 2rem; font-weight: bold; color: #ff6b6b;'>{avg_conf}</div>
                    <div style='opacity: 0.8;'>Avg Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size: 2.5rem; margin-bottom: 8px;'>📐</div>
                    <div style='font-size: 2rem; font-weight: bold; color: #f78166;'>640x640</div>
                    <div style='opacity: 0.8;'>Resolution</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown("""
                <div class='image-container'>
                    <div class='image-title'>🖼️ AI Detection Result</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(ann, use_container_width=True)
                
                img_bytes = image_to_bytes(ann)
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                st.download_button("📥 Download Result (PNG)", data=img_bytes, file_name=f"acne-detections-{ts}.png", mime="image/png", use_container_width=True)
                
            with col_b:
                st.markdown("""
                <div class='image-container'>
                    <div class='image-title'>📋 Detection Details</div>
                </div>
                """, unsafe_allow_html=True)
                df = detections_to_df(dets)
                if not df.empty:
                    st.dataframe(df.sort_values("confidence", ascending=False), use_container_width=True, hide_index=True, height=200)
                else:
                    st.info("No detections found")
                
                json_str = json.dumps(st.session_state.detections, indent=2)
                st.download_button("📥 Download Data (JSON)", data=json_str, file_name=f"acne-detections-{ts}.json", mime="application/json", use_container_width=True)

                # Bagian Rekomendasi - langsung di kolom kanan untuk mengisi ruang kosong
                if dets:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<h4 style='color: #3fb950; margin-bottom: 15px;'>💊 Treatment Recommendations</h4>", unsafe_allow_html=True)
                    
                    # Ringkas jumlah per kelas
                    counts: Dict[str, int] = {}
                    for d in dets:
                        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

                    # Tampilkan pills kompak
                    for cls, n in counts.items():
                        st.markdown(f"<div style='display: inline-block; padding: 4px 12px; margin: 2px; border-radius: 15px; background: linear-gradient(45deg, #3fb950, #58a6ff); color: white; font-size: 0.8rem; font-weight: 600;'>{cls.title()} • {n}</div>", unsafe_allow_html=True)

                    recos_by_class, unique_treatments = summarize_recommendations(dets)
                    
                    # Tampilkan rekomendasi kompak
                    st.markdown("<br>", unsafe_allow_html=True)
                    for cls, reco in recos_by_class.items():
                        with st.expander(f"💡 {cls.title()}", expanded=False):
                            for t in reco["treatments"]:
                                st.markdown(f"• {t}")
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, rgba(255, 193, 7, 0.15), rgba(255, 152, 0, 0.1)); 
                                        border-left: 4px solid #ffc107; 
                                        border-radius: 8px; 
                                        padding: 12px; 
                                        margin: 10px 0; 
                                        font-size: 0.85rem; 
                                        color: #e8e6e3;
                                        backdrop-filter: blur(5px);'>
                                <strong style='color: #ffc107;'>📝 Note:</strong><br>
                                {reco['note']}
                            </div>
                            """, unsafe_allow_html=True)

                    # Ringkasan lengkap
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**📦 Summary:**")
                    st.markdown("""
                    <div style='background: rgba(63, 185, 80, 0.1); border: 1px solid rgba(63, 185, 80, 0.3); border-radius: 8px; padding: 10px; font-size: 0.9rem;'>
                    """, unsafe_allow_html=True)
                    for t in unique_treatments:
                        st.markdown(f"• {t}")
                    st.markdown("</div>", unsafe_allow_html=True)

    with tab_about:
        st.write("**Model:** YOLO (Ultralytics) with automatic preprocessing")
        st.write("**Preprocessing:** Resize 640x640 + pixel normalization 0-1")
        st.write("**Disclaimer:** This application is only an initial suggestion, not a substitute for medical consultation.")


if __name__ == "__main__":
    streamlit_app()
