import yaml
import os
from pathlib import Path
from singleton_decorator import singleton


@singleton
class ConfigManager:
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent
        self.config_path = os.path.join(self.base_path, 'config')

    def load_config(self, config_type='live'):
        file_name = f'{config_type}.yaml'
        file_path = os.path.join(self.config_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def update_live_config(self, new_params):
        config = self.load_config('live')
        config['strategy']['parameters'].update(new_params)
        file_path = os.path.join(self.config_path, 'live.yaml')
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, sort_keys=False)
