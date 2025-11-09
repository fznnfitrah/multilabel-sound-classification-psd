import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import tsfel
import os
import io
import soundfile as sf
from audiorecorder import audiorecorder
from pydub import AudioSegment

# =========================================
# 1. KONFIGURASI & FUNGSI UTILITAS
# =========================================
st.set_page_config(page_title="Klasifikasi Audio Multilabel", page_icon="🎙️")

@st.cache_resource
def load_all_components():
    """Memuat semua komponen ML dari file .pkl"""
    try:
        components = {
            "model": joblib.load("saved_models/17 Fitur/random_forest_audio_multilabel(17).pkl"),
            "scaler": joblib.load("saved_models/17 Fitur/scaler_audio_multilabel(17).pkl"),
            "pca": joblib.load("saved_models/17 Fitur/pca_selected_audio_multilabel(17).pkl"),
            "le": joblib.load("saved_models/17 Fitur/label_encoder_audio_multilabel(17).pkl"),
            "imputer": joblib.load("saved_models/17 Fitur/imputer_audio_multilabel(17).pkl"),
            "features": joblib.load("saved_models/17 Fitur/selected_features_17feature.pkl")
        }
        return components
    except Exception as e:
        st.error(f"Gagal memuat komponen: {e}. Pastikan folder 'saved_models' lengkap.")
        return None

def extract_features_from_signal(signal, sr):
    """Ekstraksi fitur TSFEL langsung dari sinyal audio (numpy array)"""
    try:
        cfg = tsfel.get_features_by_domain()
        # Pastikan sinyal adalah 1D array (mono)
        if len(signal.shape) > 1:
             signal = np.mean(signal, axis=1)
        features = tsfel.time_series_features_extractor(cfg, signal, fs=sr, verbose=0)
        return features
    except Exception as e:
        st.error(f"Error ekstraksi fitur: {e}")
        return None

def process_audio(audio_data, sr, comps):
    """Pipeline lengkap: Ekstraksi -> Filter IG -> Imputasi -> Scaling -> PCA -> Prediksi"""
    # 1. Ekstraksi Fitur
    raw_features_df = extract_features_from_signal(audio_data, sr)
    if raw_features_df is not None:
        # 2. Filter sesuai fitur IG saat training
        filtered_df = raw_features_df.reindex(columns=comps["features"], fill_value=0)
        # 3. Preprocessing lanjutan
        imputed = comps["imputer"].transform(filtered_df)
        scaled = comps["scaler"].transform(imputed)
        pca_data = comps["pca"].transform(scaled)
        # 4. Prediksi
        pred_idx = comps["model"].predict(pca_data)
        pred_label = comps["le"].inverse_transform(pred_idx)[0]
        return pred_label
    return None

# =========================================
# 2. HALAMAN UTAMA
# =========================================
st.title("🎙️ Demo: Klasifikasi Perintah Suara")
st.write("Gunakan perekam di bawah ATAU unggah file untuk tes.")

comps = load_all_components()

if comps:
    # --- TAB UNTUK PILIHAN INPUT ---
    tab1, tab2 = st.tabs(["🔴 Rekam Langsung", "📁 Unggah File"])

    # --- TAB 1: REKAM LANGSUNG ---
    with tab1:
        st.subheader("Rekam Suara Anda")
        # Widget perekam audio
        audio = audiorecorder("Klik untuk merekam", "Klik untuk stop")

        if len(audio) > 0:
            # Putar hasil rekaman
            st.audio(audio.export().read())

            if st.button("🔍 Analisis Rekaman"):
                with st.spinner("Memproses rekaman..."):
                    # Konversi audio bytes ke numpy array
                    # audiorecorder mengembalikan objek pydub.AudioSegment
                    # Kita perlu konversi ke format yang bisa dibaca librosa/sf
                    
                    # Cara 1: Simpan sementara lalu load dengan librosa (paling aman)
                    audio.export("temp_record.wav", format="wav")
                    signal, sr = librosa.load("temp_record.wav", sr=None)
                    
                    # Proses prediksi
                    pred_label = process_audio(signal, sr, comps)
                    if pred_label:
                        # Tampilkan hasil
                        parts = pred_label.split('_')
                        identitas = parts[0] if len(parts) > 0 else "?"
                        aksi = parts[1] if len(parts) > 1 else "?"
                        
                        st.divider()
                        c1, c2 = st.columns(2)
                        c1.metric("👤 Pembicara", identitas.title())
                        c2.metric("📢 Perintah", aksi.upper())
                    
                    # Bersihkan file temp
                    if os.path.exists("temp_record.wav"):
                        os.remove("temp_record.wav")

    # --- TAB 2: UNGGAH FILE ---
    with tab2:
        st.subheader("Unggah File Audio")
        uploaded_file = st.file_uploader("", type=["wav", "mp3", "aac"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("🔍 Analisis File"):
                with st.spinner("Memproses file..."):
                    try:
                        # 1. Simpan file yang diunggah sementara ke disk
                        # Kita ambil ekstensi file aslinya (misal: .aac)
                        file_ext = uploaded_file.name.split('.')[-1]
                        temp_input_path = f"temp_input.{file_ext}"
                        
                        with open(temp_input_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # 2. Gunakan Pydub untuk konversi ke WAV (format standar paling aman)
                        # Pydub lebih kuat menangani berbagai format seperti AAC
                        audio = AudioSegment.from_file(temp_input_path)
                        temp_wav_path = "temp_converted.wav"
                        audio.export(temp_wav_path, format="wav")
                        
                        # 3. Load file WAV hasil konversi dengan Librosa
                        signal, sr = librosa.load(temp_wav_path, sr=None)
                        
                        # 4. Proses prediksi
                        pred_label = process_audio(signal, sr, comps)
                        
                        if pred_label:
                            parts = pred_label.split('_')
                            identitas = parts[0] if len(parts) > 0 else "?"
                            aksi = parts[1] if len(parts) > 1 else "?"
                            
                            st.divider()
                            c1, c2 = st.columns(2)
                            c1.metric("👤 Pembicara", identitas.title())
                            c2.metric("📢 Perintah", aksi.upper())
                        
                        # 5. Bersihkan file sementara
                        if os.path.exists(temp_input_path):
                            os.remove(temp_input_path)
                        if os.path.exists(temp_wav_path):
                            os.remove(temp_wav_path)

                    except Exception as e:
                        st.error(f"Gagal memproses audio: {e}")
                        # Tetap coba bersihkan file temp jika terjadi error
                        if os.path.exists("temp_input.*"): # Wildcard sederhana untuk jaga-jaga
                            pass
else:
    st.warning("Menunggu file model di folder 'saved_models'...")