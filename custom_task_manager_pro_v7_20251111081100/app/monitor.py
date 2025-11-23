import threading, time, psutil
from collections import deque

class SystemMonitor(threading.Thread):
    def __init__(self, interval=2, maxlen=300):
        super().__init__(daemon=True)
        self.interval = interval
        self.cpu = deque(maxlen=maxlen)
        self.mem = deque(maxlen=maxlen)
        self.net_sent = deque(maxlen=maxlen)
        self.net_recv = deque(maxlen=maxlen)
        self.disk = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)
        self._last_net = psutil.net_io_counters()

    def sample(self):
        t = time.time()
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        net = psutil.net_io_counters()
        try:
            disk = psutil.disk_usage('/').percent
        except Exception:
            disk = 0
        sent = max(0, net.bytes_sent - self._last_net.bytes_sent)
        recv = max(0, net.bytes_recv - self._last_net.bytes_recv)
        self._last_net = net
        self.cpu.append(cpu); self.mem.append(mem); self.net_sent.append(sent); self.net_recv.append(recv); self.disk.append(disk); self.timestamps.append(t)

    def run(self):
        while True:
            self.sample(); time.sleep(self.interval)

    def overview(self):
        return {'cpu': round(self.cpu[-1],2) if self.cpu else 0,
                'mem': round(self.mem[-1],2) if self.mem else 0,
                'disk': round(self.disk[-1],2) if self.disk else 0,
                'net_recv': int(self.net_recv[-1]) if self.net_recv else 0,
                'net_sent': int(self.net_sent[-1]) if self.net_sent else 0,
                'procs': len(psutil.pids())}
