# models/whisper_model.py
from root_path import setup_python_path
setup_python_path()

from settings import Settings

import whisper
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
from settings import Settings
import os
from base_model import ASRModel

@dataclass
class TranscriptionMetrics:
    timestamp: str
    model_size: str
    device: str
    compute_type: str
    audio_duration: float
    processing_time: float
    peak_memory: float
    language: str

@dataclass
class StreamingMetrics:
    timestamp: str
    first_text_latency: float
    median_latency: float
    max_latency: float
    total_audio_duration: float
    total_processing_time: float
    chunks_processed: int
    empty_chunks: int
    peak_memory: float

class WhisperModel(ASRModel):
    def __init__(self, 
                 model_size: str = "large-v2",
                 device: Optional[str] = None,
                 compute_type: str = "float16",
                 use_vad: bool = True,
                 stream: bool = True):         
        super().__init__(device)

        """
        Initialize Whisper model with specified configuration
        
        Args:
            model_size: Size of Whisper model ("tiny", "base", "small", "medium", "large-v2")
            device: Computing device ("cuda", "cpu", "mps", or None for auto-detection)
            compute_type: Type of computation ("float16" or "float32")
            use_vad: Whether to use Voice Activity Detection
        """
        self.model_size = model_size
        self.compute_type = compute_type
        self.sampling_rate = 16000
        self.use_vad = use_vad
        self.model = None
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
                self.compute_type = "float32"
                device = "cpu"
        
        if device == "cpu":
            self.compute_type = "float32"
        
        self.device = device
        
        
        # Initialize VAD if requested
        if self.use_vad:
            try:
                self.vad_model, self.vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )

                self.vad_model = self.vad_model.to(self.device)
            except Exception as e:
                self.logger.warning(f"Failed to load VAD model: {e}")
                self.use_vad = False

    def load_model(self) -> None:
        """Load and optimize Whisper model"""
        try:
            model_path = Settings.MODELS_DIR / self.model_size
            
            if self.device == "mps":
                self.model = whisper.load_model(
                    self.model_size,
                    device="cpu",
                    download_root=str(model_path)
                ).to(torch.device("mps"))
            else:
                self.model = whisper.load_model(
                    self.model_size,
                    device=self.device,
                    download_root=str(model_path)
                )
                
                if self.device == "cuda" and self.compute_type == "float16":
                    self.model = self.model.half()
            
            self.logger.info(f"Loaded {self.model_size} model on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise

    @torch.no_grad()
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file with metrics
        
        Args:
            audio_path: Path to audio file or numpy array
            **kwargs: Additional arguments for whisper.transcribe
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.perf_counter()
        peak_memory = 0

        try:
            # Load audio
            if isinstance(audio_path, str):
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
                audio = whisper.load_audio(audio_path)
            elif isinstance(audio_path, np.ndarray) and len(audio_path.shape) == 1:
                audio = audio_path
            elif isinstance(audio_path, torch.Tensor) and len(audio_path.shape) == 1:
                audio = audio_path
            else:
                raise ValueError("Invalid audio input. Must be a file path or a 1D NumPy array.")
            
            audio_duration = len(audio) / self.sampling_rate

            # Track memory for CUDA
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            
            # Set default parameters for Hindi
            params = {
                "language": "hi",
                "task": "transcribe",
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                **kwargs
            }
            
            result = self.model.transcribe(audio, **params)
            
            # Collect metrics
            end_time = time.perf_counter()
            if self.device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            metrics = TranscriptionMetrics(
                timestamp=datetime.now().isoformat(),
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                audio_duration=audio_duration,
                processing_time=end_time - start_time,
                peak_memory=peak_memory,
                language=result.get("language", "hi")
            )
            
            # Save metrics
            self._save_metrics(metrics, "transcription")
            
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"],
                "metrics": asdict(metrics),
                "processing_time": end_time - start_time,
                "audio_duration": audio_duration,
            }
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for real-time audio input.
        """
        if status:
            self.logger.warning(f"Stream callback status: {status}")
        
        # Convert stereo to mono if necessary
        if indata.shape[1] > 1:
            indata = np.mean(indata, axis=1)

        # Convert to float32 (required for Whisper)
        indata = indata.astype(np.float32)

        # Overlap handling: Append the last portion of the previous chunk
        overlap_samples = int(self.sampling_rate * self.overlap_duration)
        if len(self.audio_buffer) > 0:
            previous_chunk = self.audio_buffer[-1][-overlap_samples:]
            indata = np.concatenate((previous_chunk, indata), axis=0)

        # Add new chunk to the buffer
        self.audio_buffer.append(indata)

        # Prevent buffer from growing indefinitely
        if len(self.audio_buffer) > 10:
            self.audio_buffer.popleft()

    
    def process_audio_chunks(self, stream_start_time):
        """
        Processes audio chunks from the buffer, transcribes them, and yields results.
        """
        latencies = []
        chunks_processed = 0
        empty_chunks = 0
        first_text_time = None

        while len(self.audio_buffer) > 0:
            chunk = self.audio_buffer.popleft()
            chunk_start = time.perf_counter()

            # Run VAD if enabled
            if not self.use_vad or self._detect_speech(chunk):
                result = self.model.transcribe(
                    chunk,
                    language="hi",
                    task="transcribe",
                    temperature=0.0,
                    beam_size=5
                )

                chunk_latency = time.perf_counter() - chunk_start
                latencies.append(chunk_latency)
                chunks_processed += 1

                if result["text"].strip():
                    if first_text_time is None:
                        first_text_time = time.perf_counter() - stream_start_time

                    yield {
                        "text": result["text"],
                        "is_final": True,
                        "latency": chunk_latency,
                        "first_text_latency": first_text_time
                    }
                else:
                    empty_chunks += 1
            else:
                empty_chunks += 1
                yield {"text": "", "is_final": True}


    def stream(self, chunk_duration=2.0, overlap_duration=0.5):
        """
        Performs real-time streaming transcription with overlap handling.

        Args:
            chunk_duration (float): Duration of each chunk in seconds.
            overlap_duration (float): Overlap duration between chunks in seconds.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Store chunk & overlap size
        self.chunk_samples = int(self.sampling_rate * chunk_duration)
        self.overlap_duration = overlap_duration
        self.audio_buffer.clear()

        stream_start_time = time.perf_counter()

        try:
            with sd.InputStream(samplerate=self.sampling_rate, channels=1,
                                callback=self.audio_callback, blocksize=self.chunk_samples):
                self.logger.info("Started streaming...")

                while True:
                    yield from self.process_audio_chunks(stream_start_time)

        except KeyboardInterrupt:
            self.logger.info("Streaming stopped by user")
            self._save_streaming_metrics(stream_start_time)
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            raise

    def _detect_speech(self, audio: np.ndarray, threshold: float = 0.5) -> bool:
        """VAD helper function"""
        if not self.use_vad:
            return True
            
        with torch.no_grad():
            vad_input = torch.from_numpy(audio).float().to(self.device)
            vad_input = vad_input / torch.max(torch.abs(vad_input))
            speech_prob = self.vad_model(vad_input, self.sampling_rate).item()
        
        return speech_prob >= threshold

    def _save_metrics(self, metrics: Union[TranscriptionMetrics, StreamingMetrics], mode: str):
        """Save metrics to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dict = asdict(metrics)
        
        # Save JSON
        # json_path = Settings.METRICS_DIR / f"{mode}_metrics_{timestamp}.json"
        # with open(json_path, 'w') as f:
        #     json.dump(metrics_dict, f, indent=2)
        
        # Save CSV
        csv_path = Settings.METRICS_DIR / f"{mode}_{self.model_size}_{self.streaming}_metrics.csv"
        df = pd.DataFrame([metrics_dict])
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    
    
    def _save_streaming_metrics(self):
        """
        Saves streaming performance metrics.
        """
        metrics = StreamingMetrics(
            timestamp=datetime.now().isoformat(),
            first_text_latency=self.first_text_time or 0.0,
            median_latency=np.median(self.latencies) if self.latencies else 0.0,
            max_latency=max(self.latencies) if self.latencies else 0.0,
            total_audio_duration=self.chunks_processed * self.chunk_duration,
            total_processing_time=sum(self.latencies),
            chunks_processed=self.chunks_processed,
            empty_chunks=self.empty_chunks,
            peak_memory=torch.cuda.max_memory_allocated() / 1024**3 if self.device == "cuda" else 0
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = Settings.METRICS_DIR / f"{self.model_size}_whisper_streaming_metrics_{timestamp}.json"
        csv_path = Settings.METRICS_DIR / f"{self.model_size}_whisper_streaming_metrics.csv"

        with open(json_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

        df = pd.DataFrame([asdict(metrics)])
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    # Example usage
    model = WhisperModel(
        model_size="large-v2",
        use_vad=True,
        metrics_dir="./metrics"
    )
    model.load_model()
    
    # Non-streaming example
    result = model.transcribe("")
    print(f"Non-streaming result: {result['text']}")
    print(f"Metrics: {result['metrics']}")
    
    # Streaming example
    try:
        for result in model.stream(chunk_duration=2.0):
            if result["text"]:
                print(f"\rTranscription: {result['text']}", end="")
                print(f"\nChunk latency: {result['latency']:.3f}s")
    except KeyboardInterrupt:
        print("\nStreaming stopped")
