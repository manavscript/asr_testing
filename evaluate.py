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
import torch
from models.whisper_model import WhisperModel
from models.wave2vec2 import Wav2Vec2HindiASR  
from models.sarvam import SarvamAIASR
from utils.measurements import compute_hindi_cer, compute_hindi_wer
import torchaudio
from torchaudio.transforms import Resample
from pydub import AudioSegment
from utils.preprocess_audio import load_audio_with_librosa, preprocess_audio, process_audio, process_audio_array
from tqdm import tqdm

hf_token = Settings.HF_TOKEN
Settings.setup_directories()

# Ensure results directory exists
os.makedirs(Settings.RESULTS_DIR, exist_ok=True)

# Define available datasets
DATASETS = {
    "Common Voice": lambda: load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="test", streaming=True, trust_remote_code=True),
    # "KathBath": lambda: load_dataset("ai4bharat/kathbath", name = "hi", split="test", language="hi", streaming=True, trust_remote_code=True),
    # "IndicST": lambda: load_dataset("ai4bharat/IndicVoices-ST", "indic2en", split="hindi", streaming=True, trust_remote_code=True),
    "IndicSuperb": lambda: load_dataset("collabora/indic-superb", split="test", streaming=True, trust_remote_code=True)
}


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

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = device

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
            # sampled_data = random.sample(list(dataset), min(self.num_samples, len(dataset)))
            dataset_results = []
            sample_num = 1

            for sample in tqdm(dataset, total=self.num_samples):
                print(sample)

                if "sentence" in sample:
                    ref_text = sample["sentence"]
                elif "text" in sample:
                    ref_text = sample["text"]
                elif "transcription" in sample:
                    ref_text = sample["transcription"]
                else:
                    print("Missing ref text")
                    raise LookupError("Cannot find ref text")
                
                audio_data = {}

                if "audio" in sample:
                    audio_path = sample["audio"]["path"]
                    audio_data = sample["audio"]
                elif "chunked_audio_filepath" in sample:
                    audio_path = sample["chunked_audio_filepath"]["path"]
                    audio_data = sample["chunked_audio_filepath"]
                else:
                    print("Missing audio path")
                    raise LookupError("Cannot audio path")

                # Load and preprocess audio
                data_audio = process_audio_array(audio_data["array"], orig_sr = audio_data["sampling_rate"], target_sr=16000)
                audio, sr = data_audio["array"], data_audio["sampling_rate"]

                # Get ASR transcription
                if self.device == "cuda" and self.model == WhisperModel:
                    audio = torch.tensor(audio, dtype=torch.float16, device=self.device)  # Convert to float16
                else:
                    audio = torch.tensor(audio, dtype=torch.float32, device=self.device)
                
                pred_text = self.model.transcribe(audio)

                # Compute WER and CER
                wer_score = compute_hindi_wer(ref_text, pred_text["text"])
                cer_score = compute_hindi_cer(ref_text, pred_text["text"])

                result_entry = {
                    "Model": self.model.__class__.__name__,
                    "Dataset": dataset_name,
                    "WER": wer_score,
                    "CER": cer_score,
                    "VAD": self.vad,
                    "Streaming": self.streaming,
                    "Reference": ref_text,
                    "Prediction": pred_text["text"],
                    "Audio Path": audio_path,
                    "Latency": pred_text["processing_time"],
                    "Duration": pred_text["audio_duration"]
                }

                dataset_results.append(result_entry)
                all_results.append(result_entry)

                print(f"[{self.model.__class__.__name__}] ({dataset_name}) REF: {ref_text} | PRED: {pred_text['text']} | WER: {wer_score:.2f} | CER: {cer_score:.2f}")

                if sample_num == self.num_samples:
                    break
                else:
                    sample_num += 1

            # Save individual dataset results
            os.makedirs(os.path.join(self.results_dir, self.model.__class__.__name__), exist_ok=True)

            dataset_filename = os.path.join(self.results_dir, self.model.__class__.__name__, f"{dataset_name.replace(' ', '_').lower()}_{timestamp}.csv")
            pd.DataFrame(dataset_results).to_csv(dataset_filename, index=False)

        # Save summary results
        os.makedirs(os.path.join(self.results_dir, self.model.__class__.__name__), exist_ok=True)
        summary_filename = os.path.join(self.results_dir,  self.model.__class__.__name__, f"summary_results_{timestamp}.csv")
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
    parser.add_argument("--model", type=str, default="whisper", choices=["whisper", "wave2vec2", "sarvam"], help="Choose ASR model")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per dataset")
    parser.add_argument("--vad", action="store_true", help="Enable Voice Activity Detection (VAD)")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")

    args = parser.parse_args()

    # Initialize Whisper model
    if args.model == "whisper":
        asr_model = WhisperModel(use_vad=args.vad, model_size="medium", stream=args.streaming)
    elif args.model == "wave2vec2":
        asr_model = Wav2Vec2HindiASR(use_vad=args.vad)
    elif args.model == "sarvam":
        asr_model = SarvamAIASR(language_code="hi-IN")

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
