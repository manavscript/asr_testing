from pathlib import Path
import sys

def get_project_root() -> Path:
    """Get absolute path to project root"""
    return Path(__file__).parent.absolute()

def setup_python_path():
    """Add project root to Python path"""
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
