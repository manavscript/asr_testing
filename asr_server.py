from fastapi import FastAPI, UploadFile, File, Form
import torch
import time
import numpy as np
import whisper
import soundfile as sf
from pydub import AudioSegment  # Import Pydub for conversion
from models.whisper_model import WhisperModel
from models.wave2vec2 import Wav2Vec2HindiASR
from models.sarvam import SarvamAIASR
import io

app = FastAPI()

# Load ASR Models
AVAILABLE_MODELS = {
    "Whisper-Medium": WhisperModel(model_size="medium", device="cuda" if torch.cuda.is_available() else "cpu"),
    "Wave2Vec2": Wav2Vec2HindiASR(),
    "Sarvam": SarvamAIASR(language_code="hi-IN")
}

# Load models at startup
for model in AVAILABLE_MODELS.values():
    model.load_model()

@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...), 
    model_name: str = Form(...)
):
    if model_name not in AVAILABLE_MODELS:
        return {"error": f"Model '{model_name}' not found. Available models: {list(AVAILABLE_MODELS.keys())}"}

    asr_model = AVAILABLE_MODELS[model_name]
    
    try:

        # Convert M4A to WAV if needed
        if file.filename.endswith(".m4a"):
            # st.write("🔄 Converting M4A to WAV...")
            audio_bytes = await file.read()
            audio_buffer = io.BytesIO(audio_bytes)
            audio = AudioSegment.from_file(audio_buffer, format="m4a")
            audio = audio.set_frame_rate(16000).set_channels(1)  # Ensure 16kHz & Mono

            # Convert to WAV
            temp_wav = io.BytesIO()
            audio.export(temp_wav, format="wav")
            temp_wav.seek(0)

            # Read audio into NumPy array
            audio_data, sr = sf.read(temp_wav, dtype="float32")

        else:
            audio_data, sr = sf.read(file.file, dtype="float32")
        
        if sr != 16000:
            from scipy.signal import resample_poly
            audio_data = resample_poly(audio_data, 16000, sr)
            sr = 16000

        # Ensure mono audio
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono

        # Run ASR
        start_time = time.time()
        result = asr_model.transcribe(audio_data)

        end_time = time.time()
        print(result)
        
        latency = round(end_time - start_time, 3)

        return {
            "text": result["text"],
            "latency": latency,
            "model": model_name
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
