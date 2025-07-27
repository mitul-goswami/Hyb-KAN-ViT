import os
import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._update(config_dict)
    
    def _update(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def load_config(config_name: str, config_dir: str = None) -> Config:
    """Load configuration from YAML file
    
    Args:
        config_name: Name of config file without extension (e.g., 'base')
        config_dir: Directory containing config files (default: './configs')
    
    Returns:
        Config object with nested attributes
    """
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__))
    
    config_path = os.path.join(config_dir, f"{config_name}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)

try:
    base_config = load_config("base")
except FileNotFoundError:
    base_config = None

__all__ = ['Config', 'load_config', 'base_config']
