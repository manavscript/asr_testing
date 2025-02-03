from root_path import setup_python_path
setup_python_path()

from settings import Settings

import torch
import time
import json
import numpy as np
from typing import Dict, Any, Optional, Generator, Union
from dataclasses import dataclass, asdict
from collections import deque
import sounddevice as sd
from datetime import datetime
import pandas as pd
import logging
import os
from base_model import ASRModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

@dataclass
class TranscriptionMetrics:
    timestamp: str
    model_name: str
    device: str
    audio_duration: float
    processing_time: float
    peak_memory: float

class Wav2Vec2HindiASR(ASRModel):
    def __init__(self, 
                 model_name: str = "theainerd/Wav2Vec2-large-xlsr-hindi",
                 device: Optional[str] = None,
                 use_vad: bool = True,
                 stream: bool = True):         
        super().__init__(device)

        self.model_name = model_name
        self.sampling_rate = 16000
        self.use_vad = use_vad
        self.model = None
        self.processor = None
        self.audio_buffer = deque(maxlen=5)
        self.streaming = stream
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        
        # Load model
        self.load_model()

    def load_model(self) -> None:
        """Load Wav2Vec2 model and processor"""
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
            self.logger.info(f"Loaded {self.model_name} model on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load Wav2Vec2 model: {e}")
            raise

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file with Wav2Vec2 model
        
        Args:
            audio_path: Path to audio file
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.perf_counter()
        peak_memory = 0

        try:
            # Load and preprocess audio
            # if not os.path.exists(audio_path):
            #     raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
            
            # waveform, sample_rate = torchaudio.load(audio_path)

            if isinstance(audio_path, str):
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
                waveform, sample_rate = torchaudio.load_audio(audio_path)
            elif isinstance(audio_path, np.ndarray) and len(audio_path.shape) == 1:
                waveform, sample_rate = audio_path, 16000
                waveform = waveform.squeeze()
            elif isinstance(audio_path, torch.Tensor) and len(audio_path.shape) == 1:
                waveform, sample_rate = audio_path, 16000
                waveform = waveform.squeeze().numpy()
            else:
                raise ValueError("Invalid audio input. Must be a file path or a 1D NumPy array.")

            if sample_rate != self.sampling_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)(waveform)
                waveform = waveform.squeeze().numpy()

            # Track memory for CUDA
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            # Process audio
            input_values = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            # Collect metrics
            end_time = time.perf_counter()
            if self.device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

            metrics = TranscriptionMetrics(
                timestamp=datetime.now().isoformat(),
                model_name=self.model_name,
                device=self.device,
                audio_duration=len(waveform) / self.sampling_rate,
                processing_time=end_time - start_time,
                peak_memory=peak_memory
            )
            
            self._save_metrics(metrics)
            
            return {
                "text": transcription,
                "metrics": asdict(metrics),
                "processing_time": end_time - start_time,
                "audio_duration": end_time - start_time,
            }
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise

    def _save_metrics(self, metrics: TranscriptionMetrics):
        """Save transcription metrics to CSV"""
        csv_path = Settings.METRICS_DIR / f"{self.model_name}_transcription_metrics.csv"
        df = pd.DataFrame([asdict(metrics)])
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    model = Wav2Vec2HindiASR()
    result = model.transcribe("path/to/audio.wav")
    print(f"Transcription: {result['text']}")
    print(f"Metrics: {result['metrics']}")
