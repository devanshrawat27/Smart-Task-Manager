import os, joblib
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'categorizer.joblib')

def heuristic(name):
    n = (name or '').lower()
    if 'chrome' in n or 'firefox' in n: return 'browser'
    if 'python' in n or 'node' in n: return 'script'
    if 'svchost' in n or 'system' in n: return 'system'
    if 'vlc' in n or 'spotify' in n: return 'media'
    if 'mysql' in n or 'mongod' in n: return 'database'
    if 'code' in n or 'pycharm' in n: return 'ide'
    if 'onedrive' in n or 'backup' in n: return 'utility'
    if 'steam' in n or 'valorant' in n: return 'game'
    if 'defender' in n or 'kaspersky' in n: return 'security'
    return 'other'

class Categorizer:
    def __init__(self):
        self.model = None
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
            except Exception as e:
                print('Failed to load categorizer model:', e)

    def predict(self, info):
        if not info or (not self.model):
            return heuristic(info.get('name','') if info else '')
        try:
            X = [[
                info.get('cpu',0),
                info.get('memory',0),
                info.get('threads',0),
                info.get('io_read',0),
                info.get('io_write',0),
                info.get('ctx_vol',0),
                info.get('ctx_invol',0),
                info.get('uptime',0)
            ]]
            pred = self.model.predict(X)[0]
            return pred
        except Exception:
            return heuristic(info.get('name',''))
