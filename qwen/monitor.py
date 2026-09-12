"""Bounded Linux/WSL resource sampling, including desktop VRAM usage."""
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import threading
import time


def counters(path):
    values={}
    for line in Path(path).read_text().splitlines():
        parts=line.replace(':','').split()
        if len(parts)>1 and parts[1].isdigit():
            values[parts[0]]=int(parts[1])*(1024 if len(parts)>2 and parts[2]=='kB' else 1)
    return values


class Monitor:
    def __init__(self,path,interval=1.0,pid=None):
        self.path=Path(path);self.interval=interval;self.pid=pid or os.getpid()
        self.stop=threading.Event();self.phase='startup';self.samples=[];self.start=time.monotonic()

    def sample(self):
        result=dict(seconds=time.monotonic()-self.start,phase=self.phase,
                    utc=datetime.now(timezone.utc).isoformat())
        try:
            status=counters(f'/proc/{self.pid}/status');mem=counters('/proc/meminfo');vm=counters('/proc/vmstat')
            for name in ('VmRSS','VmHWM','VmSwap'):result[name]=status.get(name)
            for name in ('MemAvailable','SwapFree','SwapTotal'):result[name]=mem.get(name)
            for name in ('pswpin','pswpout'):result[name]=vm.get(name)
        except FileNotFoundError:result['process_exited']=True
        except (OSError,ValueError) as exc:result['host_error']=str(exc)
        try:
            run=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,memory.free,memory.reserved,utilization.gpu',
                                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
            used,total,free,reserved,util=(int(x.strip()) for x in run.stdout.splitlines()[0].split(','))
            result.update(gpu_used_bytes=used*1024**2,gpu_total_bytes=total*1024**2,gpu_free_bytes=free*1024**2,
                          gpu_reserved_bytes=reserved*1024**2,gpu_unavailable_bytes=(used+reserved)*1024**2,gpu_util_percent=util)
        except (OSError,ValueError,subprocess.SubprocessError) as exc:result['gpu_error']=str(exc)
        self.samples.append(result)
        self.stream.write(json.dumps(result)+'\n');self.stream.flush()

    def loop(self):
        while not self.stop.wait(self.interval):self.sample()

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.stream=self.path.open('w')
        self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.sample();self.stream.close()

    def summary(self):
        def maximum(key):return max((s[key] for s in self.samples if s.get(key) is not None),default=None)
        def minimum(key):return min((s[key] for s in self.samples if s.get(key) is not None),default=None)
        def change(key):
            low,high=minimum(key),maximum(key)
            return high-low if low is not None and high is not None else None
        return dict(sample_interval_seconds=self.interval,samples=len(self.samples),
            peak_rss_bytes=maximum('VmRSS'),process_high_water_rss_bytes=maximum('VmHWM'),peak_swap_bytes=maximum('VmSwap'),
            min_host_available_bytes=minimum('MemAvailable'),peak_gpu_used_bytes=maximum('gpu_used_bytes'),
            min_gpu_free_bytes=minimum('gpu_free_bytes'),peak_gpu_reserved_bytes=maximum('gpu_reserved_bytes'),
            peak_gpu_unavailable_bytes=maximum('gpu_unavailable_bytes'),
            gpu_total_bytes=maximum('gpu_total_bytes'),
            swap_in_pages=change('pswpin'),swap_out_pages=change('pswpout'),
            errors=sum('host_error' in s or 'gpu_error' in s for s in self.samples),
            gpu_measurement='NVML whole device; used and driver-reserved memory are separate, free is queried directly; sampled peaks can miss short spikes')
