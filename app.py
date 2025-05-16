import streamlit as st
import pickle
import numpy as np

# Load model, vectorizer, dan label encoder dari file dictionary tunggal
with open('svm_ovr_model.pkl', 'rb') as f:
    data = pickle.load(f)
    models = data['models']
    tfidf = data['tfidf']
    le = data['label_encoder']

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Berita", layout="centered", page_icon="📰")

# Styling dengan tema gelap elegan
st.markdown("""
    <style>
        body {
            background-color: #1e1e2f;
        }
        .main {
            background-color: #2c2f48;
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        .stTextArea textarea {
            background-color: #3a3f5c !important;
            color: white !important;
            font-size: 16px !important;
        }
        .result-badge {
            display: inline-block;
            padding: 0.4em 0.8em;
            background-color: #00a8ff;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton button {
            background-color: #00a8ff;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Judul halaman
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("📰 Klasifikasi Berita Otomatis")
st.subheader("📢 Masukkan teks berita dan dapatkan kategorinya 🔍")
st.markdown("Sistem ini akan memprediksi kategori dari teks berita yang Anda masukkan berdasarkan model klasifikasi SVM yang telah dilatih.")

# Layout dua kolom
col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area("📝 Masukkan Teks Berita Anda", height=200)

with col2:
    st.markdown("### 📌 Petunjuk:")
    st.markdown("- Gunakan teks berita asli.")
    st.markdown("- Minimal beberapa kalimat agar hasil akurat.")
    st.markdown("- Klik tombol di bawah untuk klasifikasi.")

st.markdown("")

if st.button("🔍 Klasifikasikan"):
    if not input_text.strip():
        st.warning("⚠️ Silakan masukkan teks berita terlebih dahulu.")
    else:
        # Transformasi input menggunakan TF-IDF
        X_input = tfidf.transform([input_text]).toarray()

        # Hitung skor untuk masing-masing model OvR
        scores = [np.dot(X_input[0], model[0]) + model[1] for model in models]

        # Prediksi label berdasarkan skor tertinggi
        predicted_label = np.argmax(scores)
        final_label = le.inverse_transform([predicted_label])[0]

        # Tampilkan hasil
        st.markdown("### ✅ Hasil Klasifikasi:")
        st.markdown(f"<div class='result-badge'>{final_label}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
