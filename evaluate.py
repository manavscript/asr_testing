from root_path import setup_python_path
setup_python_path()

from settings import Settings

import librosa
import numpy as np
import random
import argparse
import pandas as pd
import os
from datetime import datetime
from datasets import load_dataset
from jiwer import wer, cer
from settings import Settings  # Importing settings for RESULTS_DIR

# Ensure results directory exists
os.makedirs(Settings.RESULTS_DIR, exist_ok=True)

# Define available datasets
DATASETS = {
    "Common Voice": lambda: load_dataset("mozilla-foundation/common_voice_15_0", "hi", split="test"),
    "KathBath": lambda: load_dataset("kathbath/hindi_speech_dataset", split="test"),
    "IndicTTS": lambda: load_dataset("iisc/IndicTTS", "hi", split="test")
}

# Convert speech to 16kHz mono audio
def preprocess_audio(audio_path):
    """
    Load an audio file and convert it to 16kHz mono format.
    Args:
        audio_path: Path to the audio file.
    Returns:
        waveform: 1D NumPy array of audio samples.
        sr: Sampling rate (should be 16kHz).
    """
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    return waveform, sr

class Evaluator:
    def __init__(self, model, num_samples=50, vad=False, streaming=False):
        """
        Initializes the ASR evaluator.
        Args:
            model: An ASR model instance that implements `transcribe(audio)`.
            num_samples: Number of test samples per dataset.
            vad: Whether VAD was used.
            streaming: Whether streaming mode was used.
        """
        self.model = model
        self.num_samples = num_samples
        self.vad = vad
        self.streaming = streaming
        self.results_dir = Settings.RESULTS_DIR

    def evaluate(self):
        """
        Runs evaluation on all datasets.
        Saves results to CSV files.
        Returns:
            Dictionary with model name, average WER, and average CER.
        """
        all_results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for dataset_name, load_func in DATASETS.items():
            dataset = load_func()
            sampled_data = random.sample(list(dataset), min(self.num_samples, len(dataset)))
            dataset_results = []

            for sample in sampled_data:
                ref_text = sample["sentence"]
                audio_path = sample["path"]

                # Load and preprocess audio
                audio, sr = preprocess_audio(audio_path)

                # Get ASR transcription
                pred_text = self.model.transcribe(audio)

                # Compute WER and CER
                wer_score = wer(ref_text, pred_text)
                cer_score = cer(ref_text, pred_text)

                result_entry = {
                    "Model": self.model.__class__.__name__,
                    "Dataset": dataset_name,
                    "WER": wer_score,
                    "CER": cer_score,
                    "VAD": self.vad,
                    "Streaming": self.streaming,
                    "Reference": ref_text,
                    "Prediction": pred_text,
                    "Audio Path": audio_path
                }

                dataset_results.append(result_entry)
                all_results.append(result_entry)

                print(f"[{self.model.__class__.__name__}] ({dataset_name}) REF: {ref_text} | PRED: {pred_text} | WER: {wer_score:.2f} | CER: {cer_score:.2f}")

            # Save individual dataset results
            dataset_filename = os.path.join(self.results_dir, f"{dataset_name.replace(' ', '_').lower()}_{timestamp}.csv")
            pd.DataFrame(dataset_results).to_csv(dataset_filename, index=False)

        # Save summary results
        summary_filename = os.path.join(self.results_dir, f"summary_results_{timestamp}.csv")
        pd.DataFrame(all_results).to_csv(summary_filename, index=False)

        avg_wer = np.mean([r["WER"] for r in all_results])
        avg_cer = np.mean([r["CER"] for r in all_results])

        return {
            "Model": self.model.__class__.__name__,
            "Average WER": avg_wer,
            "Average CER": avg_cer,
            "VAD": self.vad,
            "Streaming": self.streaming
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate an ASR model on benchmarks.")
    parser.add_argument("--model", type=str, default="whisper", choices=["whisper"], help="Choose ASR model")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per dataset")
    parser.add_argument("--vad", action="store_true", help="Enable Voice Activity Detection (VAD)")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")

    args = parser.parse_args()

    # Import the Whisper model dynamically
    from models.whisper_model import WhisperModel  

    # Initialize Whisper model
    asr_model = WhisperModel(use_vad=args.vad, model_size="medium")
    asr_model.load_model()

    # Initialize evaluator
    evaluator = Evaluator(model=asr_model, num_samples=args.num_samples, vad=args.vad, streaming=args.streaming)

    # Run evaluation
    results = evaluator.evaluate()

    print("\n🔹 Evaluation Results:")
    print(f"📊 Model: {results['Model']}")
    print(f"📊 Average WER: {results['Average WER']:.2f}")
    print(f"📊 Average CER: {results['Average CER']:.2f}")
    print(f"🔹 VAD Enabled: {results['VAD']}")
    print(f"🔹 Streaming Enabled: {results['Streaming']}")
    print(f"📁 Results saved in: {Settings.RESULTS_DIR}")

if __name__ == "__main__":
    main()
