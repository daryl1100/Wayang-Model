import streamlit as st
import numpy as np
from PIL import Image

# ------- load TF/Keras models --------
import tensorflow as tf
import keras
keras_backend = "keras"
import os; os.environ["KERAS_BACKEND"] = keras_backend
eff_model = tf.keras.models.load_model("models/wayang_efficientnetv2s.keras")
mob_model = tf.keras.models.load_model("models/wayang_mobilenetv3large.keras")

# ------- load PyTorch model ----------
import torch, timm
device = "cpu"
classes = ["Abimanyu", "Antasena", "Arjuna", "Bagong", "Bima", "Cepot", "Gareng",
    "Gatot Kaca", "Hanoman", "Kresna", "Nakula", "Petruk", "Semar", "Yudhistira"
]

deit = timm.create_model("deit_small_patch16_224", pretrained=False,
                         num_classes=len(classes))
deit.load_state_dict(torch.load("models/wayang_deit_small.pth", map_location=device))
deit.eval()

# ------- helper for TF models --------
def preprocess_tf(img, size=224):
    img = img.resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    return arr[np.newaxis, ...]

# ------- helper for PyTorch ----------
import torchvision.transforms as T
pt_tf = T.Compose([
    T.Resize(224), T.CenterCrop(224),
    T.ToTensor(),  T.Normalize([0.5]*3,[0.5]*3)
])

def predict_pytorch(img):
    with torch.no_grad():
        out = deit(pt_tf(img).unsqueeze(0)).softmax(1)[0]
    idx = out.argmax().item()
    return classes[idx], float(out[idx])

# ---------------- Streamlit UI ----------------
st.set_page_config(
    page_title="Wayang Classification",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS untuk styling elegan
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Poppins:wght@300;400;500;600&display=swap');
    
    /* Global styles */
    body {
        background: linear-gradient(135deg, #f9f7f7 0%, #e6f0ff 100%);
        font-family: 'Poppins', sans-serif;
        color: #333;
        scroll-behavior: smooth;
    }
    
    /* Navbar styling - Enhanced */
    .navbar {
        background: linear-gradient(90deg, #1a2a6c 0%, #2a3c7f 50%, #1a2a6c 100%);
        overflow: hidden;
        border-radius: 10px;
        margin: 0 auto 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        max-width: 900px;
        display: flex;
        justify-content: center;
        position: sticky;
        top: 10px;
        z-index: 100;
    }
    .navbar ul {
        list-style-type: none;
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
    }
    .navbar li {
        float: left;
        margin: 0 5px;
    }
    .navbar li a {
        display: block;
        color: #fff;
        text-align: center;
        padding: 18px 28px;
        text-decoration: none;
        font-size: 18px;
        font-weight: 500;
        transition: all 0.3s ease;
        position: relative;
        border-radius: 8px;
    }
    .navbar li a:hover {
        color: #ffd700;
        background: rgba(255, 255, 255, 0.1);
    }
    .navbar li a.active {
        color: #ffd700 !important;
        background: rgba(255, 255, 255, 0.15) !important;
    }
    .navbar li a.active:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 80%;
        height: 3px;
        background: #d4af37;
        transform: translateX(-50%);
    }
    
    /* Header styling */
    .header {
        font-family: 'Playfair Display', serif;
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #1a2a6c !important;
        text-align: center;
        padding: 30px;
        margin-bottom: 30px;
        position: relative;
        letter-spacing: 1px;
    }
    .header:after {
        content: '';
        display: block;
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #d4af37, #ffd700);
        margin: 15px auto;
        border-radius: 2px;
    }
    
    /* Card styling */
    .card {
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        background: white;
        margin-bottom: 25px;
        border-top: 4px solid #d4af37;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    .card:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1a2a6c, #2a3c7f);
    }
    .model-name {
        font-size: 20px;
        font-weight: 600;
        color: #1a2a6c;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .model-name i {
        font-size: 24px;
        color: #d4af37;
    }
    .prediction {
        font-size: 26px;
        font-weight: 700;
        color: #2a3c7f;
        margin: 15px 0;
        font-family: 'Playfair Display', serif;
    }
    .confidence {
        font-size: 16px;
        color: #5a6a8c;
        margin-top: 5px;
        font-weight: 500;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1a2a6c, #2a3c7f);
        border-radius: 4px;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #a5b4d4;
        border-radius: 15px;
        padding: 40px 30px;
        text-align: center;
        background: rgba(255, 255, 255, 0.7);
        margin-bottom: 30px;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    .upload-area:hover {
        border-color: #d4af37;
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* Portrait card styling for wayang characters */
    .portrait-card {
        display: flex;
        flex-direction: column;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        height: 100%;
        background: white;
    }
    .portrait-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    .portrait-img {
        width: 100%;
        aspect-ratio: 3/4;
        object-fit: cover;
        border-bottom: 3px solid #d4af37;
    }
    .portrait-content {
        padding: 20px;
        text-align: center;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .portrait-title {
        font-family: 'Playfair Display', serif;
        font-size: 22px;
        font-weight: 700;
        color: #1a2a6c;
        margin-bottom: 8px;
    }
    .portrait-subtitle {
        font-size: 16px;
        color: #5a6a8c;
        font-style: italic;
    }
    
    /* Premium settings card styling */
    .premium-settings {
        padding: 40px 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        margin-bottom: 35px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(212, 175, 55, 0.3);
        z-index: 1;
    }
    .premium-settings:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #d4af37, #ffd700);
        z-index: 2;
    }
    .premium-settings:after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("https://www.transparenttextures.com/patterns/light-wool.png");
        opacity: 0.1;
        z-index: -1;
    }
    
    /* Section styling */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 32px;
        font-weight: 700;
        color: #1a2a6c;
        text-align: center;
        margin: 40px 0 25px;
        position: relative;
    }
    .section-title:after {
        content: '';
        display: block;
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, #d4af37, #ffd700);
        margin: 10px auto;
        border-radius: 2px;
    }
    
    /* Footer styling */
    .footer {
        padding: 25px;
        text-align: center;
        background: linear-gradient(90deg, #1a2a6c 0%, #2a3c7f 100%);
        color: white;
        border-radius: 10px 10px 0 0;
        margin-top: 50px;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1a2a6c, #2a3c7f) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-weight: 500 !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 15px rgba(26, 42, 108, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 7px 20px rgba(26, 42, 108, 0.4) !important;
    }
    
    /* Custom multiselect */
    .stMultiSelect [data-baseweb=select] {
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        border: 1px solid #d0d9f0 !important;
        background: white !important;
    }
    
    /* Icon styling */
    .icon-large {
        font-size: 36px;
        color: #d4af37;
        margin-bottom: 15px;
    }
    
    /* Model badge styling */
    .model-badge {
        display: inline-flex;
        align-items: center;
        padding: 8px 15px;
        border-radius: 50px;
        background: rgba(212, 175, 55, 0.15);
        color: #1a2a6c;
        font-weight: 500;
        margin: 5px;
    }
    
    /* Anchor styling */
    .section-anchor {
        position: absolute;
        top: -100px;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .navbar li a {
            padding: 14px 16px;
            font-size: 16px;
        }
        .header {
            font-size: 32px !important;
            padding: 20px;
        }
        .portrait-card {
            margin-bottom: 20px;
        }
        .navbar ul {
            flex-wrap: wrap;
        }
        .navbar li {
            margin: 5px;
        }
    }
    </style>
    
    <!-- Font Awesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown('<div class="header">Klasifikasi Tokoh Wayang</div>', unsafe_allow_html=True)

# Navbar with proper anchors
st.markdown("""
<div class="navbar">
    <ul>
        <li><a class="active" href="#header"><i class="fas fa-home"></i> Beranda</a></li>
        <li><a href="#tentang"><i class="fas fa-book"></i> Tentang Wayang</a></li>
        <li><a href="#model"><i class="fas fa-brain"></i> Model</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main container
main_container = st.container()

# Fungsi prediksi untuk model
def predict(model_name, img):
    if model_name == "EfficientNetV2S (Keras)":
        pred = eff_model.predict(preprocess_tf(img))[0]
        idx = pred.argmax()
        return classes[idx], float(pred[idx])
    elif model_name == "MobileNetV3Large (Keras)":
        pred = mob_model.predict(preprocess_tf(img))[0]
        idx = pred.argmax()
        return classes[idx], float(pred[idx])
    elif model_name == "DeiT-Small (PyTorch)":
        return predict_pytorch(img)

# Tampilkan konten utama
with main_container:
    # Anchor for header
    st.markdown('<div id="header" class="section-anchor"></div>', unsafe_allow_html=True)
    
    # Penjelasan Wayang di bagian atas
    st.markdown('<div id="tentang" class="section-title">Apa itu Wayang?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: justify; margin-bottom: 30px; padding: 20px; background: white; border-radius: 15px; box-shadow: 0 8px 30px rgba(0,0,0,0.08);'>
        <p style="font-size: 17px; line-height: 1.8;">
        <i class="fas fa-quote-left" style="color:#d4af37; font-size:24px; margin-right:10px;"></i>
        Wayang adalah seni pertunjukan asli Indonesia yang berkembang pesat di Pulau Jawa dan Bali. 
        Pertunjukan ini menggunakan boneka atau figur yang dimainkan oleh seorang dalang. Wayang 
        tidak hanya sekedar pertunjukan hiburan, tetapi juga mengandung nilai-nilai filosofis, 
        pendidikan, dan spiritual yang dalam. Cerita wayang umumnya diambil dari epik Hindu seperti 
        Mahabharata dan Ramayana, serta siklus cerita Panji.
        </p>
        <p style="font-size: 17px; line-height: 1.8; margin-top: 15px;">
        UNESCO telah mengakui wayang sebagai Masterpiece of Oral and Intangible Heritage of Humanity 
        pada tahun 2003. Wayang memiliki berbagai jenis seperti wayang kulit (terbuat dari kulit kerbau), 
        wayang golek (boneka kayu), dan wayang orang (dimainkan langsung oleh manusia).
        </p>
        <div style="text-align: center; margin-top: 25px;">
            <i class="fas fa-award" style="color:#d4af37; font-size:28px;"></i>
            <p style="font-style: italic; color:#5a6a8c;">Diakui oleh UNESCO sebagai Warisan Budaya Dunia</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Contoh gambar tokoh wayang - Portrait cards
    st.markdown('<div class="section-title">Tokoh Wayang Indonesia</div>', unsafe_allow_html=True)
    
    # Daftar tokoh wayang dengan deskripsi singkat
    wayang_characters = [
        {
            "name": "Arjuna",
            "image": "https://mediaindonesia.gumlet.io/news/2018/09/edcdd878e00303eda2d124a0f2788d7a.jpg?w=360&dpr=2.6",
            "description": "Ksatria Pandawa ahli panah"
        },
        {
            "name": "Bima",
            "image": "https://static.promediateknologi.id/crop/0x0:0x0/0x0/webp/photo/p2/214/2024/07/29/Sambut-Tahun-Baru-2018-DPRD-DIY-Gelar-Wayang-Kulit-dengan-Lakon-Banjaran-Bima-Star-Jogja-FM-2003932352.jpeg",
            "description": "Ksatria Pandawa paling kuat"
        },
        {
            "name": "Semar",
            "image": "https://upload.wikimedia.org/wikipedia/commons/3/3c/Wayang_Kulit_of_Semar_crop.jpg",
            "description": "Penasihat bijak para ksatria"
        },
        {
            "name": "Gatot Kaca",
            "image": "https://static.vecteezy.com/system/resources/thumbnails/052/323/450/small_2x/wayang-puppet-shadow-gatotkaca-image-illustration-javanese-traditional-performance-art-free-vector.jpg",
            "description": "Ksatria bersayap anak Bima"
        }
    ]
    
    # Membuat 4 kolom untuk gambar (2 rows of 4)
    cols = st.columns(4)
    
    for i, character in enumerate(wayang_characters):
        with cols[i % 4]:
            # Portrait card for wayang character
            st.markdown(f'''
            <div class="portrait-card">
                <img src="{character['image']}" class="portrait-img" alt="{character['name']}">
                <div class="portrait-content">
                    <div class="portrait-title">{character['name']}</div>
                    <div class="portrait-subtitle">{character['description']}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Garis pemisah sebelum bagian klasifikasi
    st.markdown("---")
    
    # Section title untuk klasifikasi
    st.markdown('<div id="model" class="section-title">Klasifikasi Tokoh Wayang</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-size:18px; color:#5a6a8c; margin-bottom:30px;">'
                'Gunakan alat ini untuk mengidentifikasi tokoh wayang dari gambar'
                '</p>', unsafe_allow_html=True)
    
    # Premium settings card untuk pengaturan model
    st.markdown('<div class="premium-settings">', unsafe_allow_html=True)
    
    # Icon and title
    st.markdown('<div style="text-align:center; margin-bottom:25px;">'
                '<div style="display:inline-block; background:#1a2a6c; width:80px; height:80px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 20px;">'
                '<i class="fas fa-microchip" style="color:#ffd700; font-size:36px;"></i>'
                '</div>'
                '<h2 style="color:#1a2a6c; font-family:\'Playfair Display\', serif; font-size:32px; margin-bottom:10px;">Pengaturan Model</h2>'
                '<p style="color:#5a6a8c; max-width:600px; margin:0 auto;">Pilih model deep learning untuk mengidentifikasi tokoh wayang dalam gambar</p>'
                '</div>', unsafe_allow_html=True)
    
    # Model badges
    st.markdown('<div style="text-align:center; margin-bottom:25px;">'
                '<span class="model-badge"><i class="fas fa-bolt"></i> EfficientNetV2S</span>'
                '<span class="model-badge"><i class="fas fa-mobile-alt"></i> MobileNetV3Large</span>'
                '<span class="model-badge"><i class="fas fa-project-diagram"></i> DeiT-Small</span>'
                '</div>', unsafe_allow_html=True)
    
    # Model selection in a centered container with 2 columns
    col1, col2 = st.columns([1, 2])
    with col2:
        model_choice = st.multiselect(
            "Pilih Model Klasifikasi:",
            ["EfficientNetV2S (Keras)", "MobileNetV3Large (Keras)", "DeiT-Small (PyTorch)"],
            default=["EfficientNetV2S (Keras)"]
        )
    
    st.markdown('<div style="text-align:center; margin-top:20px;">'
                '<i class="fas fa-info-circle" style="color:#d4af37;"></i>'
                '<span style="color:#5a6a8c; margin-left:8px;">Pilih minimal satu model untuk melakukan klasifikasi</span>'
                '</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close premium settings
    
    # Upload section
    st.markdown('<div class="section-title" style="font-size:28px;">Unggah Gambar Wayang</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Pilih gambar wayang (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        
        # Display image and predictions in columns
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="section-title" style="font-size:24px; margin:0 0 15px;">Gambar Input</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True, caption="Gambar yang diunggah")
        
        with col2:
            if model_choice:
                st.markdown('<div class="section-title" style="font-size:24px; margin:0 0 15px;">Hasil Prediksi</div>', unsafe_allow_html=True)
                
                # Vertical layout for predictions
                for model_name in model_choice:
                    with st.spinner(f"Memproses {model_name}..."):
                        label, conf = predict(model_name, img)
                        
                        # Get model icon
                        if "EfficientNet" in model_name:
                            icon = "fas fa-bolt"
                        elif "MobileNet" in model_name:
                            icon = "fas fa-mobile-alt"
                        else:
                            icon = "fas fa-project-diagram"
                        
                        st.markdown(f'<div class="card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="model-name"><i class="{icon}"></i> {model_name}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="prediction">{label}</div>', unsafe_allow_html=True)
                        
                        # Progress bar untuk confidence score
                        st.progress(conf, text=f"Confidence: {conf*100:.2f}%")
                        
                        # Interpretasi confidence score
                        if conf > 0.9:
                            st.success("Prediksi sangat yakin")
                        elif conf > 0.7:
                            st.info("Prediksi cukup yakin")
                        else:
                            st.warning("Prediksi kurang yakin")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Silakan pilih minimal satu model untuk klasifikasi")
    else:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.markdown('<div style="display:inline-block; background:#1a2a6c; width:100px; height:100px; border-radius:50%; display:flex; align-items:center; justify-content:center; margin:0 auto 20px;">'
                    '<i class="fas fa-cloud-upload-alt" style="color:#ffd700; font-size:48px;"></i>'
                    '</div>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#1a2a6c;'>Unggah gambar untuk memulai klasifikasi</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#5a6a8c;'>Format yang didukung: JPG, PNG, JPEG</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer elegan
st.markdown("""
<div class="footer">
    <p style="margin:0; font-size:16px;">
        Aplikasi Klasifikasi Tokoh Wayang © 2023 | 
        Dibangun dengan Streamlit, TensorFlow/Keras, dan PyTorch
    </p>
</div>
""", unsafe_allow_html=True)

# JavaScript untuk navigasi navbar
st.markdown("""
<script>
// Fungsi untuk scroll ke anchor dengan offset
function scrollToAnchor(anchorId) {
    const anchor = document.getElementById(anchorId);
    if (anchor) {
        const offset = 100; // Offset untuk navbar
        const position = anchor.getBoundingClientRect().top + window.pageYOffset - offset;
        window.scrollTo({ top: position, behavior: 'smooth' });
    }
}

// Tangani klik navbar
document.querySelectorAll('.navbar a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        
        // Update active class
        document.querySelectorAll('.navbar a').forEach(a => a.classList.remove('active'));
        this.classList.add('active');
        
        // Scroll ke target
        scrollToAnchor(targetId);
    });
});

// Update active state on scroll
window.addEventListener('scroll', function() {
    const sections = ['header', 'tentang', 'model'];
    let currentSection = '';
    
    sections.forEach(section => {
        const element = document.getElementById(section);
        if (element) {
            const rect = element.getBoundingClientRect();
            if (rect.top <= 150 && rect.bottom >= 150) {
                currentSection = section;
            }
        }
    });
    
    // Update active navbar item
    document.querySelectorAll('.navbar a').forEach(a => {
        a.classList.remove('active');
        if (a.getAttribute('href') === '#' + currentSection) {
            a.classList.add('active');
        }
    });
});
</script>
""", unsafe_allow_html=True)