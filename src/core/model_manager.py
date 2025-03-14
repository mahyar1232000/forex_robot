import hashlib
from datetime import datetime
from tensorflow.keras.models import load_model


class ModelVersioner:
    def save_model(self, model, params):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        params_hash = hashlib.md5(str(params).encode()).hexdigest()[:8]
        filename = f"model_{timestamp}_{params_hash}.h5"

        model.save(f"models/versioned_models/{filename}")
        self._update_model_registry(filename, params)

    def load_latest_model(self):
        registry = self._load_model_registry()
        latest = sorted(registry.items(), key=lambda x: x[1]['timestamp'])[-1]
        return load_model(latest[0]), latest[1]['params']
