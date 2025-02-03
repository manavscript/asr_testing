import streamlit as st
import requests
import tempfile
import sounddevice as sd
import numpy as np
import soundfile as sf
import time

API_URL = "http://localhost:8000/transcribe/"  # ASR API Endpoint

st.title("🎤 Real-time ASR with Model Selection")
st.write("Upload an audio file (MP3, WAV, M4A) or **record your voice** to transcribe.")

# Step 1: Select ASR Model
model_choice = st.selectbox("🛠 Select ASR Model:", ["Whisper-Medium", "Wave2Vec2", "Sarvam"])

# Step 2: Choose Input Method
option = st.radio("📂 Select Input Method:", ["Upload Audio", "🎙️ Record Audio"])

audio_path = None  # To store recorded or uploaded audio

# Step 3: Upload Audio (No Processing in UI)
if option == "Upload Audio":
    uploaded_file = st.file_uploader("📂 Upload an audio file", type=["wav", "mp3", "m4a"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name  # Send raw file

        st.success("✅ Audio file ready for transcription!")

# Step 4: Record Audio (Still Ensuring 16kHz)
elif option == "🎙️ Record Audio":
    duration = st.slider("⏳ Recording duration (seconds)", min_value=1, max_value=10, value=5)
    
    if st.button("🎤 Start Recording"):
        st.write("🎙️ Recording... Speak now!")

        # Record audio (Ensure 16kHz & Mono)
        audio_data = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()  # Wait until recording is finished

        st.write("✅ Recording complete!")

        # Save recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, audio_data, 16000)
            audio_path = tmp_file.name  # Assign temp file to audio_path
            st.success("✅ Audio successfully recorded!")

# Step 5: Transcribe Audio (Send Raw File to API)
if audio_path is not None and st.button("🚀 Transcribe"):
    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"model_name": model_choice}
        start_time = time.time()
        response = requests.post(API_URL, files=files, data=data)
        end_time = time.time()

    if response.status_code == 200:
        result = response.json()
        print(result)
        st.subheader("📜 Transcription:")
        st.write(result["text"])

        st.subheader("⏱️ Processing Latency:")
        st.write(f"⏳ {result['latency']} seconds")

        st.subheader("⚙️ ASR Model Used:")
        st.write(f"🛠 {result['model']}")

    else:
        st.error("❌ Failed to process audio!")
