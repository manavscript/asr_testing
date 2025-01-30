from pathlib import Path
from root_path import get_project_root

class Settings:
    # Base paths
    ROOT = get_project_root()
    MODELS_DIR = ROOT / "models"
    METRICS_DIR = ROOT / "metrics"
    CONFIG_DIR = ROOT / "config"
    RESULTS_DIR = ROOT / "results"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for path in [cls.MODELS_DIR, cls.METRICS_DIR, cls.CONFIG_DIR]:
            path.mkdir(parents=True, exist_ok=True)
