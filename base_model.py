# model.py
from abc import ABC, abstractmethod
import torch
import time
from typing import Dict, Any, Optional, Union
import numpy as np

class ASRModel(ABC):
    def __init__(self, device: Optional[str] = None):
        if torch.backends.mps.is_available():
            device = "mps"  # Use MPS for Apple Silicon
        elif torch.cuda.is_available():
            device = "cuda"  # Use CUDA for NVIDIA GPUs
        else:
            device = "cpu"  # Default to CPU

        self.device = device
        self.model = None
        self.processor = None
        self.sampling_rate = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor"""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file to text
        Returns dict with at least 'text' key and optional metadata
        """
        pass

    def stream(self, audio_chunk: Union[np.ndarray, bytes], **kwargs) -> Dict[str, Any]:
        """
        Process streaming audio input
        Default implementation treats it as a single chunk
        Override for true streaming support
        """
        raise NotImplementedError("Streaming not implemented for this model")

    def measure_latency(self, func):
        """Decorator to measure processing time"""
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            if isinstance(result, dict):
                result['latency'] = end_time - start_time
            else:
                result = {
                    'text': result,
                    'latency': end_time - start_time
                }
            return result
        return wrapper

    def __str__(self):
        return self.__class__.__name__