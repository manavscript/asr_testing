import librosa
import os
import torch
import torchaudio

import numpy as np

# Convert speech to 16kHz mono audio
def preprocess_audio(audio_path, target_sr):
    """
    Load an audio file and convert it to 16kHz mono format.
    Args:
        audio_path: Path to the audio file.
    Returns:
        waveform: 1D NumPy array of audio samples.
        sr: Sampling rate (should be 16kHz).
    """
    waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return waveform, sr

def process_audio(example, sr):
    audio_tensor = torch.tensor(example["audio"]["array"])  # Convert to tensor
    orig_sr = example["audio"]["sampling_rate"]

    # Ensure the audio is mono (convert stereo to mono if needed)
    if audio_tensor.ndim > 1 and audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)  # Convert stereo to mono

    # Resample if needed
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sr)
        audio_tensor = resampler(audio_tensor)

    return {"audio": {"array": audio_tensor.numpy(), "sampling_rate": sr}}


def load_audio_with_librosa(filename, target_sr=16000):
    """
    Loads an audio file (including MP3) using librosa, ensures it is mono, and resamples to target_sr.

    Args:
        filename (str): Path to the audio file.
        target_sr (int): Target sample rate (default: 16kHz).

    Returns:
        dict: Processed audio with "array" as a NumPy array and updated "sampling_rate".
    """
    # Check if file exists
    if not os.path.isfile(filename):
        print(f"❌ Error: File not found - {filename}")
        return None

    try:
        # Load audio with librosa (handles MP3, WAV, FLAC, OGG)
        waveform, orig_sr = librosa.load(filename, sr=target_sr, mono=True)  # Ensure mono

        return {"audio": {"array": waveform, "sampling_rate": target_sr}}

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return None

def process_audio_array(audio_array, orig_sr, target_sr=16000):
    """
    Processes a raw audio array: ensures it is mono and resamples to target_sr.

    Args:
        audio_array (np.ndarray): Raw audio waveform (1D NumPy array).
        orig_sr (int): Original sample rate.
        target_sr (int): Desired sample rate (default: 16kHz).

    Returns:
        dict: Processed audio with "array" as a NumPy array and updated "sampling_rate".
    """
    # Ensure mono (convert stereo if needed)
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=0)  # Convert stereo to mono

    # Resample if needed
    if orig_sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)

    return {"array": audio_array, "sampling_rate": target_sr}
