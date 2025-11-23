import os, time, psutil, threading, csv, subprocess
from flask import render_template, jsonify, request, send_file, Response, make_response
from . import app
from .collector import Collector
from .monitor import SystemMonitor
from pathlib import Path
import logging, pandas as pd, io, joblib, json
from collections import defaultdict, deque

LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_DIR/'actions.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# start collector (writes behavioral logs)
collector = Collector(interval=3)
collector.start()

# start system monitor for system charts
monitor = SystemMonitor(interval=2)
monitor.start()
app.config['monitor'] = monitor

# per-process recent history (in-memory) for sparkline
process_history = defaultdict(lambda: deque(maxlen=60))

# ML helpers
from .categorizer import Categorizer
from .predictor import LifetimePredictor
from .anomaly_detector import AnomalyDetector

categorizer = Categorizer()
predictor = LifetimePredictor()
anomaly = AnomalyDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/overview')
def overview():
    mon = app.config.get('monitor')
    if not mon:
        return jsonify({'cpu':0,'mem':0,'disk':0,'net_recv':0,'net_sent':0,'procs':0})
    return jsonify(mon.overview())

@app.route('/api/processes')
def processes():
    procs = []
    for p in psutil.process_iter(['pid','name','username','cpu_percent','memory_percent','num_threads','create_time']):
        try:
            info = p.info
            pid = info.get('pid')
            uptime = 0
            try:
                uptime = int(time.time() - info.get('create_time', time.time()))
            except:
                uptime = 0
            cpu = info.get('cpu_percent') or 0
            mem = round(info.get('memory_percent') or 0,2)
            threads = info.get('num_threads') or 0
            # append to in-memory history for sparkline
            try:
                process_history[pid].append(float(cpu))
            except Exception:
                pass
            # minimal io/ctx retrieval (best-effort)
            io_read = io_write = ctx_vol = ctx_invol = 0
            try:
                io = p.io_counters(); io_read = getattr(io,'read_bytes',0); io_write = getattr(io,'write_bytes',0)
            except Exception: pass
            try:
                ctx = p.num_ctx_switches(); ctx_vol = getattr(ctx,'voluntary',0); ctx_invol = getattr(ctx,'involuntary',0)
            except Exception: pass
            info_dict = {'name': info.get('name'), 'cpu': cpu, 'memory': mem, 'threads': threads,
                         'io_read': io_read, 'io_write': io_write, 'ctx_vol': ctx_vol, 'ctx_invol': ctx_invol, 'uptime': uptime}
            cat = categorizer.predict(info_dict)
            a_score = anomaly.score(info_dict)
            # lifetime will be hidden in main table; included in detail endpoint
            procs.append({'pid': pid, 'name': info.get('name'), 'user': info.get('username'),
                          'cpu': cpu, 'memory': mem, 'threads': threads, 'category': cat, 'anomaly': a_score})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception as e:
            logging.exception('Process iterate error: %s', e)
            continue
    procs = sorted(procs, key=lambda x: x['cpu'], reverse=True)
    return jsonify(procs)

@app.route('/api/process/<int:pid>')
def process_detail(pid):
    # find process and build detailed info including lifetime and history
    try:
        p = psutil.Process(pid)
        info = p.as_dict(attrs=['pid','name','username','cpu_percent','memory_percent','num_threads','create_time'], ad_value=None)
        uptime = 0
        try:
            uptime = int(time.time() - info.get('create_time', time.time()))
        except:
            uptime = 0
        cpu = info.get('cpu_percent') or 0
        mem = round(info.get('memory_percent') or 0,2)
        threads = info.get('num_threads') or 0
        io_read = io_write = ctx_vol = ctx_invol = 0
        try:
            io = p.io_counters(); io_read = getattr(io,'read_bytes',0); io_write = getattr(io,'write_bytes',0)
        except Exception: pass
        try:
            ctx = p.num_ctx_switches(); ctx_vol = getattr(ctx,'voluntary',0); ctx_invol = getattr(ctx,'involuntary',0)
        except Exception: pass
        info_dict = {'name': info.get('name'), 'cpu': cpu, 'memory': mem, 'threads': threads,
                     'io_read': io_read, 'io_write': io_write, 'ctx_vol': ctx_vol, 'ctx_invol': ctx_invol, 'uptime': uptime}
        cat = categorizer.predict(info_dict)
        life = predictor.predict(info_dict)
        a_score = anomaly.score(info_dict)
        history = list(process_history.get(pid, []))
        return jsonify({'pid':pid,'name':info.get('name'),'user':info.get('username'),'cpu':cpu,'memory':mem,'threads':threads,'uptime':uptime,'category':cat,'lifetime_pred':life,'anomaly':a_score,'history':history})
    except psutil.NoSuchProcess:
        return jsonify({'error':'no such process'}), 404
    except Exception as e:
        logging.exception('Detail error: %s', e)
        return jsonify({'error':str(e)}), 500

@app.route('/api/kill', methods=['POST'])
def kill_proc():
    data = request.get_json() or {}
    pid = int(data.get('pid', -1))
    try:
        p = psutil.Process(pid)
        name = p.name()
        p.terminate()
        gone, alive = psutil.wait_procs([p], timeout=3)
        if alive:
            p.kill()
        logging.info(f'KILLED pid={pid} name={name}')
        return jsonify({'status':'ok','msg':f'Killed {pid} - {name}'})
    except psutil.NoSuchProcess:
        return jsonify({'status':'error','msg':'No such process'}), 404
    except psutil.AccessDenied:
        return jsonify({'status':'error','msg':'Access denied - need elevated permissions'}), 403
    except Exception as e:
        return jsonify({'status':'error','msg':str(e)}), 500

@app.route('/api/export')
def export_csv():
    behavioral = os.path.join(Path(__file__).parent.parent, 'logs', 'system_data_behavioral.csv')
    if os.path.exists(behavioral):
        with open(behavioral,'r',encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/csv', headers={"Content-Disposition":"attachment; filename=system_data_behavioral.csv"})
    else:
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['timestamp','pid','name','user','cpu','memory','threads','io_read_bytes','io_write_bytes','ctx_vol','ctx_invol','create_time'])
        return make_response(si.getvalue())

# optional train/reload endpoints (kept server-side but not exposed in UI)
@app.route('/api/reload_models', methods=['POST'])
def reload_models():
    global categorizer, predictor, anomaly
    try:
        categorizer = Categorizer(); predictor = LifetimePredictor(); anomaly = AnomalyDetector()
        logging.info('Models reloaded via /api/reload_models')
        return jsonify({'status':'ok','msg':'Models reloaded'})
    except Exception as e:
        logging.exception('Failed to reload models: %s', e)
        return jsonify({'status':'error','msg':str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
