# 🎤 ASR Model Evaluation & Benchmarking

This project evaluates **automatic speech recognition (ASR) models** (e.g., Whisper) on multiple datasets, including **Common Voice**, **KathBath**, and **IndicTTS**.

## 🚀 Features
- ✅ **Supports Whisper & other ASR models**  
- ✅ **Benchmark on real-world datasets**  
- ✅ **Evaluate WER (Word Error Rate), CER (Character Error Rate)**  
- ✅ **Supports streaming & VAD (Voice Activity Detection)**  
- ✅ **Runs on local machine, remote servers, or Google Colab**  

---

## 📂 Project Structure
```
/asr_models/                   # Project root
│── /models/                   # ASR model implementations
│   ├── __init__.py            
│   ├── whisper_model.py       # Whisper ASR model
│── /evaluations/              # Evaluation scripts
│   ├── __init__.py
│   ├── evaluator.py           # Evaluates ASR models
│── /results/                  # Stores benchmark results
│── /datasets/                 # Custom datasets (if needed)
│── settings.py                # Configuration (paths, tokens, etc.)
│── run_colab.ipynb            # Colab setup
│── requirements.txt           # Dependencies
│── README.md                  # This file
│── main.py                    # Main execution file
```

---

## 💾 Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YOUR_GITHUB/asr_models.git
cd asr_models
```

### **2️⃣ Set Up a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **3️⃣ Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔧 Configuration

### **🔑 Hugging Face Token (For Common Voice 17.0)**
1. Get your **Hugging Face access token** from:  
   [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Add it as an environment variable:  
   ```bash
   export HUGGINGFACE_TOKEN="your_token_here"
   ```
3. Or save it in `settings.py`:
   ```python
   import os
   class Settings:
       HF_TOKEN = "your_token_here"
   ```

---

## 🎯 Running the ASR Evaluation

### **📝 Evaluate ASR Models**
Run:
```bash
python evaluations/evaluator.py --num_samples 50 --vad --streaming
```

**CLI Arguments:**
| Argument | Description | Example |
|----------|------------|---------|
| `--num_samples` | Number of test samples per dataset | `--num_samples 100` |
| `--vad` | Enable voice activity detection | `--vad` |
| `--streaming` | Enable streaming transcription | `--streaming` |

---

## 📊 Viewing Results

Results are saved in `/results/`:
```
results/
├── common_voice_20250129_140000.csv
├── kathbath_20250129_140000.csv
├── indictts_20250129_140000.csv
├── summary_results_20250129_140000.csv
```

| Model | Dataset | WER | CER | VAD | Streaming | Reference | Prediction |
|--------|---------|------|------|------|-----------|------------|-------------|
<!-- | Whisper | Common Voice | 8.2% | 4.5% | ✅ | ✅ | "सुप्रभात" | "सुप्रभात" | -->

---

## 🔋 Running on Google Colab

1. Open **Google Colab**
2. Run:
   ```bash
   !git clone "https://github.com/YOUR_GITHUB/asr_models.git"
   %cd asr_models
   !pip install -r requirements.txt
   ```
3. Authenticate Hugging Face:
   ```bash
   from huggingface_hub import notebook_login
   notebook_login()
   ```
4. Run:
   ```bash
   !python evaluations/evaluator.py --num_samples 50 --vad --streaming
   ```

---

## 🛠️ Troubleshooting

### **1️⃣ `OSError: PortAudio library not found` (Google Colab)**
Run:
```bash
!apt-get install -y portaudio19-dev
!pip install sounddevice
```
Then restart the runtime:
```python
import os
os._exit(00)
```

---

### **2️⃣ `DatasetNotFoundError: Dataset is gated`**
Authenticate Hugging Face:
```bash
huggingface-cli login
```
Or pass the token:
```python
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="test", use_auth_token=hf_token)
```

---

### **3️⃣ `RuntimeError: expected scalar type Float but found Half`**
Your **Whisper model is in `float16`**, but input is `float32`. Fix by converting input:
```python
if self.compute_type == "float16" and self.device == "cuda":
    audio = torch.tensor(audio, dtype=torch.float16, device=self.device)
else:
    audio = torch.tensor(audio, dtype=torch.float32, device=self.device)
```

---

## 💪 Future Improvements
- 🔹 Support for **multiple ASR models** beyond Whisper.
- 🔹 Add **parallel processing for faster evaluations**.
- 🔹 Improve **speaker diarization support**.

---

## 📄 Contact
For issues or contributions, create a pull request or open an issue on GitHub.

---