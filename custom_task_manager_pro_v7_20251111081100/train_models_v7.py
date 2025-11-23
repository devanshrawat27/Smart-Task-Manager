import os, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

LOG_PATH = os.path.join('logs','system_data_behavioral.csv')
APP_MODEL_DIR = os.path.join('app','models')
os.makedirs(APP_MODEL_DIR, exist_ok=True)

SAMPLE = [
    {'name':'chrome.exe','category':'browser','uptime':300,'cpu':5,'memory':1.2,'threads':30,'io_read':1000,'io_write':500,'ctx_vol':20,'ctx_invol':5},
    {'name':'python.exe','category':'script','uptime':50,'cpu':20,'memory':2.3,'threads':5,'io_read':200,'io_write':40,'ctx_vol':5,'ctx_invol':1},
    {'name':'svchost.exe','category':'system','uptime':10000,'cpu':1,'memory':0.8,'threads':50,'io_read':10,'io_write':5,'ctx_vol':200,'ctx_invol':10},
]

def load_data():
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        if df.shape[0] < 50:
            print('Not enough behavioral log rows, using sample data. Collected rows:', df.shape[0])
            return pd.DataFrame(SAMPLE)
        df = df.dropna()
        df['name_clean'] = df['name'].fillna('unknown').astype(str)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        except Exception:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['uptime'] = (pd.Timestamp.now() - df['timestamp']).dt.total_seconds().abs()
        df = df.rename(columns={'io_read_bytes':'io_read','io_write_bytes':'io_write'})
        return df
    else:
        print('No behavioral logs found; using sample data')
        return pd.DataFrame(SAMPLE)

def train_categorizer(df):
    X = df[['cpu','memory','threads','io_read','io_write','ctx_vol','ctx_invol','uptime']]
    if 'category' not in df.columns:
        def map_cat(n):
            n=n.lower()
            if 'chrome' in n or 'firefox' in n: return 'browser'
            if 'python' in n or 'node' in n: return 'script'
            if 'svchost' in n or 'system' in n: return 'system'
            if 'vlc' in n or 'spotify' in n: return 'media'
            if 'mysql' in n or 'mongod' in n or 'sql' in n: return 'database'
            if 'code' in n or 'pycharm' in n: return 'ide'
            if 'onedrive' in n or 'backup' in n: return 'utility'
            if 'steam' in n or 'valorant' in n: return 'game'
            if 'defender' in n or 'kaspersky' in n: return 'security'
            return 'other'
        y = df['name_clean'].apply(map_cat)
    else:
        y = df['category']
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, os.path.join(APP_MODEL_DIR,'categorizer.joblib'))
    print('Saved categorizer model to app/models/')

def train_lifetime(df):
    X = df[['uptime','cpu','memory','threads','io_read','io_write']]
    y = (df['uptime'] * 0.05 + (100 - df['cpu']) * 8 - df['memory']*2).clip(1,86400)
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X, y)
    joblib.dump(reg, os.path.join(APP_MODEL_DIR,'lifetime.joblib'))
    print('Saved lifetime model to app/models/')

def train_anomaly(df):
    X = df[['cpu','memory','threads','io_read','io_write','ctx_vol','ctx_invol']].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=42)
    iso.fit(Xs)
    decisions = iso.decision_function(Xs)
    mean_dec = float(np.mean(decisions))
    std_dec = float(np.std(decisions, ddof=1))
    joblib.dump((iso, scaler, mean_dec, std_dec), os.path.join(APP_MODEL_DIR,'anomaly.joblib'))
    print('Saved anomaly model and scaler (with mean/std) to app/models/')

if __name__ == '__main__':
    df = load_data()
    train_categorizer(df)
    train_lifetime(df)
    train_anomaly(df)
    print('All v7 models trained and saved to app/models/')
