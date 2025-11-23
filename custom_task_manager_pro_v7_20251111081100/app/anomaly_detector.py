import os, joblib, numpy as np
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'anomaly.joblib')

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.mean = 0.0
        self.std = 0.08
        if os.path.exists(MODEL_PATH):
            try:
                obj = joblib.load(MODEL_PATH)
                if isinstance(obj, (list, tuple)) and len(obj) >= 2:
                    self.model = obj[0]; self.scaler = obj[1]
                    if len(obj) >= 4:
                        try:
                            self.mean = float(obj[2]); self.std = float(obj[3])
                        except Exception:
                            pass
                else:
                    self.model = obj
            except Exception as e:
                print('Failed to load anomaly model:', e)

    def score(self, info):
        cpu = info.get('cpu',0); mem = info.get('memory',0); threads = info.get('threads',0)
        io_r = info.get('io_read',0); io_w = info.get('io_write',0)
        ctx_v = info.get('ctx_vol',0); ctx_i = info.get('ctx_invol',0)

        if not self.model:
            val = (cpu/100.0)*0.45 + (mem/100.0)*0.3 + (threads/50.0)*0.1 + (io_r+io_w)/1e6*0.15
            return int(min(100, max(0, val*100)))

        try:
            import numpy as _np
            X = _np.array([[cpu, mem, threads, io_r, io_w, ctx_v, ctx_i]])
            if self.scaler is not None:
                Xs = self.scaler.transform(X)
            else:
                Xs = X
            raw = float(self.model.decision_function(Xs)[0])
            mean = self.mean if hasattr(self, 'mean') else 0.0
            std = self.std if hasattr(self, 'std') else 0.08
            z = (raw - mean) / (std if std>1e-6 else 1.0)
            score = int(max(0, min(100, 50 - z * 25)))
            return score
        except Exception as e:
            return 0
