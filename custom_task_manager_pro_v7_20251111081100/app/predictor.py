import os, joblib
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lifetime.joblib')

class LifetimePredictor:
    def __init__(self):
        self.model = None
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
            except Exception as e:
                print('Failed to load lifetime model:', e)

    def predict(self, info):
        if not self.model:
            uptime = info.get('uptime',0)
            cpu = info.get('cpu',0)
            mem = info.get('memory',0)
            return int(max(60, min(86400, (uptime * 0.05) + (100 - cpu) * 8 - mem * 2)))
        try:
            X = [[
                info.get('uptime',0), info.get('cpu',0), info.get('memory',0), info.get('threads',0),
                info.get('io_read',0), info.get('io_write',0)
            ]]
            pred = self.model.predict(X)[0]
            return int(max(1, pred))
        except Exception:
            return int(max(60, min(86400, (info.get('uptime',0) * 0.05) + (100 - info.get('cpu',0)) * 8 - info.get('memory',0) * 2)))
