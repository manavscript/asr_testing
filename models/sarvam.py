from root_path import setup_python_path
setup_python_path()

from settings import Settings

import torch
import time
import json
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import requests
from datetime import datetime
import pandas as pd
import logging
import os
from base_model import ASRModel
import os
import tempfile
import torchaudio

@dataclass
class TranscriptionMetrics:
    timestamp: str
    model_name: str
    audio_duration: float
    processing_time: float
    response_status: int

class SarvamAIASR(ASRModel):
    def __init__(self, 
                 model_name: str = "saarika:v2",
                 language_code: str = "unknown",
                 with_timestamps: bool = False,
                 with_diarization: bool = False):
        super().__init__(device=None)

        self.api_key = os.getenv("SARVAM_AI_API_KEY")
        self.model_name = model_name
        self.language_code = language_code
        self.with_timestamps = with_timestamps
        self.with_diarization = with_diarization
        self.api_url = "https://api.sarvam.ai/speech-to-text"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> None:
        return super().load_model()
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file using Sarvam AI API
        
        Args:
            audio_path: Path to audio file
        """
        format = "wav"
        try:
            if isinstance(audio_path, str):
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
                fname = audio_path

                audio_duration = 0
            elif isinstance(audio_path, np.ndarray) and len(audio_path.shape) == 1:
                audio = audio_path
                with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                    torchaudio.save(temp_file.name, torch.Tensor(audio_path).unsqueeze(0), 16000, format=format)
                    fname = temp_file.name
                
                audio_duration = len(audio) / self.sampling_rate

            elif isinstance(audio_path, torch.Tensor) and len(audio_path.shape) == 1:
                audio = audio_path
                with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                    torchaudio.save(temp_file.name, audio_path.unsqueeze(0), 16000, format=format)
                    fname = temp_file.name
                
                audio_duration = len(audio) / self.sampling_rate
            else:
                raise ValueError("Invalid audio input. Must be a file path or a 1D NumPy array.")

            files = {
                "file": (os.path.basename(fname), open(fname, "rb"), "audio/wav")
            }
            
            start_time = time.perf_counter()
            
            payload = {
                "model": self.model_name,
                "language_code": self.language_code,
                "with_timestamps": str(self.with_timestamps).lower(),
                "with_diarization": str(self.with_diarization).lower()
            }
            headers = {"api-subscription-key": self.api_key}

            response = requests.post(self.api_url, files=files, data=payload, headers=headers)
            response_json = response.json()
            print(response_json)
            
            end_time = time.perf_counter()
            
            metrics = TranscriptionMetrics(
                timestamp=datetime.now().isoformat(),
                model_name=self.model_name,
                audio_duration=0.0,  # Placeholder, API does not return duration
                processing_time=end_time - start_time,
                response_status=response.status_code
            )
            self._save_metrics(metrics)
            
            return {
                "text": response_json.get("transcript", ""),
                "metrics": asdict(metrics),
                "status_code": response.status_code,
                "processing_time": end_time - start_time,
                "audio_duration": audio_duration
            }
        
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _save_metrics(self, metrics: TranscriptionMetrics):
        """Save transcription metrics to CSV"""
        csv_path = Settings.METRICS_DIR / f"sarvam_transcription_metrics.csv"
        df = pd.DataFrame([asdict(metrics)])
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    model = SarvamAIASR(api_key="<your-api-key>")
    result = model.transcribe("path/to/audio.wav")
    print(f"Transcription: {result['text']}")
    print(f"Metrics: {result['metrics']}")
