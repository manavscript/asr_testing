from pathlib import Path
from root_path import get_project_root
import json

class Settings:
    # Base paths
    ROOT = get_project_root()
    MODELS_DIR = ROOT / "models"
    METRICS_DIR = ROOT / "metrics"
    CONFIG_DIR = ROOT / "config"
    RESULTS_DIR = ROOT / "results"

    with open("config.json", "r") as f:
        config = json.load(f)
    HF_TOKEN = config.get("huggingface_token", "")
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        print("creating dirs")
        for path in [cls.MODELS_DIR, cls.METRICS_DIR, cls.CONFIG_DIR]:
            path.mkdir(parents=True, exist_ok=True)
