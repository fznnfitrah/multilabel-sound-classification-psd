import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import tsfel
import os
from audiorecorder import audiorecorder
from pydub import AudioSegment

# ==========================================================
# 1. KONFIGURASI
# ==========================================================
st.set_page_config(page_title="Klasifikasi Audio (Model Spesialis)", page_icon="🎙️")
st.title("🎙️ Demo: Klasifikasi Perintah Suara (Model Spesialis)")
st.write("Aplikasi ini memuat dua model AI terpisah: satu untuk mendeteksi Perintah (Buka/Tutup) dan satu untuk mendeteksi Identitas (Fikri/Fauzan).")

# Tentukan path ke model Anda
MODEL_AKSI_PATH = "saved_models/new_models/model_aksi_final.pkl"
MODEL_IDENTITAS_PATH = "saved_models/new_models/model_identitas_final.pkl"
TARGET_SR = 16000 # Pastikan sama dengan SR saat training

# ==========================================================
# 2. LOAD MODEL (Diubah untuk memuat DUA model)
# ==========================================================
@st.cache_resource
def load_components():
    """Memuat kedua paket model spesialis (Aksi dan Identitas)."""
    try:
        # Muat paket model Aksi
        package_aksi = joblib.load(MODEL_AKSI_PATH)
        
        # Muat paket model Identitas
        package_identitas = joblib.load(MODEL_IDENTITAS_PATH)

        return {
            "aksi": package_aksi,
            "identitas": package_identitas
        }
        
    except FileNotFoundError as e:
        st.error(f"Gagal memuat file model: {e}")
        st.error(f"Pastikan '{MODEL_AKSI_PATH}' DAN '{MODEL_IDENTITAS_PATH}' ada di folder yang sama dengan app.py")
        return None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


# ==========================================================
# 3. EKSTRAKSI FITUR (Tidak berubah)
# ==========================================================
def extract_features_from_signal(signal, sr):
    """Ekstrak fitur TSFEL lengkap (mentah) dari sinyal audio."""
    try:
        cfg = tsfel.get_features_by_domain()
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1) # Pastikan mono

        # Ekstrak semua fitur mentah (misal: 156+ fitur)
        feature_df = tsfel.time_series_features_extractor(cfg, signal, fs=sr, verbose=0)
        return feature_df
    except Exception as e:
        st.error(f"Error ekstraksi fitur (TSFEL): {e}")
        return None


# ==========================================================
# 4. PIPELINE PREDIKSI (Diubah untuk DUA model)
# ==========================================================
def predict_specialist(signal, sr, comps_aksi, comps_identitas):
    """
    Menjalankan dua pipeline prediksi secara paralel.
    
    Args:
        signal (np.array): Sinyal audio.
        sr (int): Sampling rate.
        comps_aksi (dict): Paket model Aksi (model, scaler, features, labels).
        comps_identitas (dict): Paket model Identitas.
        
    Returns:
        hasil_aksi (str): Prediksi final (misal: "Buka").
        hasil_identitas (str): Prediksi final (misal: "Fikri").
    """
    
    # 1. Ekstrak Fitur Mentah (Satu Kali)
    raw_features = extract_features_from_signal(signal, sr)
    if raw_features is None:
        return None, None

    hasil_aksi = "Error"
    hasil_identitas = "Error"

    # --- Pipeline A (Model Aksi) ---
    try:
        # 2a. Filter fitur Aksi (sesuai N optimal)
        features_aksi = raw_features.reindex(columns=comps_aksi["selected_features"], fill_value=0)
        # 3a. Scale Aksi
        scaled_aksi = comps_aksi["scaler"].transform(features_aksi)
        # 4a. Prediksi Aksi
        pred_idx_aksi = comps_aksi["model"].predict(scaled_aksi)[0]
        # 5a. Terjemahkan (dari dict {1: "Buka", 0: "Tutup"})
        hasil_aksi = comps_aksi["labels"][pred_idx_aksi]
    except Exception as e:
        st.error(f"Error pada pipeline Model Aksi: {e}")

    # --- Pipeline B (Model Identitas) ---
    try:
        # 2b. Filter fitur Identitas (sesuai N optimal)
        features_identitas = raw_features.reindex(columns=comps_identitas["selected_features"], fill_value=0)
        # 3b. Scale Identitas
        scaled_identitas = comps_identitas["scaler"].transform(features_identitas)
        # 4b. Prediksi Identitas
        pred_idx_identitas = comps_identitas["model"].predict(scaled_identitas)[0]
        # 5b. Terjemahkan (dari dict {1: "Fikri", 0: "Fauzan"})
        hasil_identitas = comps_identitas["labels"][pred_idx_identitas]
    except Exception as e:
        st.error(f"Error pada pipeline Model Identitas: {e}")

    return hasil_aksi, hasil_identitas

# ==========================================================
# 5. HALAMAN UTAMA
# ==========================================================
all_comps = load_components()

if all_comps is None:
    st.error("Gagal memuat komponen model. Aplikasi berhenti.")
    st.stop()
else:
    st.success("✅ Model Aksi dan Model Identitas berhasil dimuat!")

tab1, tab2 = st.tabs(["🔴 Rekam Suara", "📁 Unggah File"])

# ==========================================================
# TAB 1 - Rekam Suara (Diperbarui)
# ==========================================================
with tab1:
    st.subheader("Rekam Suara Anda")
    
    # === MODIFIKASI DI SINI ===
    # Mengganti teks agar lebih jelas seperti tombol Record/Stop
    audio = audiorecorder(
        "Tekan untuk Mulai Merekam 🔴", 
        "Merekam... Tekan untuk Berhenti ⏹️"
    )
    # =========================

    if len(audio) > 0:
        st.audio(audio.export().read())

        if st.button("🔍 Analisis Rekaman"):
            with st.spinner("Memproses rekaman..."):

                # Simpan sementara
                temp_wav = "temp_record.wav"
                audio.export(temp_wav, format="wav")

                # Load dengan librosa, pastikan SR sesuai training
                signal, sr = librosa.load(temp_wav, sr=TARGET_SR, mono=True)

                # Dapatkan DUA hasil prediksi
                pred_aksi, pred_identitas = predict_specialist(
                    signal, sr, 
                    all_comps["aksi"], 
                    all_comps["identitas"]
                )

                # Tampilkan hasil
                if pred_aksi and pred_identitas:
                    st.divider()
                    st.success("Prediksi Berhasil!")
                    col1, col2 = st.columns(2)
                    col1.metric("👤 Identitas", pred_identitas)
                    col2.metric("📢 Perintah", pred_aksi)

                os.remove(temp_wav)

# ==========================================================
# TAB 2 - Upload File (Tidak berubah)
# ==========================================================
with tab2:
    st.subheader("Unggah File Audio")
    uploaded = st.file_uploader("Pilih file", type=["wav", "mp3", "aac"])

    if uploaded is not None:
        st.audio(uploaded)

        if st.button("🔍 Analisis File"):
            with st.spinner("Memproses file..."):
                
                # Konversi ke WAV (menggunakan Pydub)
                try:
                    # Simpan file input
                    ext = uploaded.name.split('.')[-1]
                    temp_in = f"temp_input.{ext}"
                    with open(temp_in, "wb") as f:
                        f.write(uploaded.getbuffer())

                    # Konversi ke WAV
                    audio_seg = AudioSegment.from_file(temp_in)
                    temp_wav = "temp_upload.wav"
                    audio_seg.export(temp_wav, format="wav")

                    # Load dengan librosa, pastikan SR sesuai training
                    signal, sr = librosa.load(temp_wav, sr=TARGET_SR, mono=True)

                    # Dapatkan DUA hasil prediksi
                    pred_aksi, pred_identitas = predict_specialist(
                        signal, sr, 
                        all_comps["aksi"], 
                        all_comps["identitas"]
                    )

                    # Tampilkan hasil
                    if pred_aksi and pred_identitas:
                        st.divider()
                        st.success("Prediksi Berhasil!")
                        col1, col2 = st.columns(2)
                        col1.metric("👤 Identitas", pred_identitas)
                        col2.metric("📢 Perintah", pred_aksi)

                    # Bersihkan file temp
                    os.remove(temp_in)
                    os.remove(temp_wav)
                
                except Exception as e:
                    st.error(f"Gagal memproses audio: {e}")
                    # Pastikan file temp dibersihkan jika ada error
                    if 'temp_in' in locals() and os.path.exists(temp_in): 
                        os.remove(temp_in)
                    if 'temp_wav' in locals() and os.path.exists(temp_wav): 
                        os.remove(temp_wav)