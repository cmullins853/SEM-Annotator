import time
import threading
import subprocess
import os

class EnergyMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.total_energy_joules = 0.0
        self.total_time_seconds = 0.0
        self.cumulative_energy_joules = 0.0
        self.cumulative_time_seconds = 0.0
        self.thread = None
        self.lock = threading.Lock()
        
    def _get_gpu_power(self):
        """
        Returns total GPU power draw in Watts.
        Returns 0.0 if fails.
        """
        try:
            import pynvml
            if getattr(self, '_nvml_inited', None) is None:
                pynvml.nvmlInit()
                self._nvml_inited = True
            
            watts = 0.0
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                watts += power_mw / 1000.0
            return watts
            
        except Exception:
            pass # fallback to subprocess
        
        try:
            kwargs = {'encoding': 'utf-8', 'capture_output': True, 'text': True}
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                kwargs['startupinfo'] = startupinfo
                
            # query power.draw for all GPUs
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                timeout=1.0,
                check=True,
                **kwargs
            )
            # sum up power from all GPUs
            total_watts = sum(float(line.strip()) for line in result.stdout.strip().split('\n') if line.strip())
            return total_watts
        except Exception as e:
            return 0.0

    def _monitor_loop(self):
        last_time = time.time()
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            watts = self._get_gpu_power()
            
            with self.lock:
                # Energy (Joules) = Power (Watts) * Time (Seconds)
                self.total_energy_joules += watts * dt
                self.total_time_seconds += dt
                self.cumulative_energy_joules += watts * dt
                self.cumulative_time_seconds += dt
                
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("⚡ Energy Monitoring Started...")

    def stop(self):
        self.running = False
        print("⚡ Energy Monitoring Stopped.")

    def reset(self):
        with self.lock:
            self.total_energy_joules = 0.0
            self.total_time_seconds = 0.0

    def get_stats(self):
        with self.lock:
            joules = self.total_energy_joules
            seconds = self.total_time_seconds
            cumulative_joules = self.cumulative_energy_joules
            cumulative_seconds = self.cumulative_time_seconds
        
        kwh = joules / 3.6e6
        cumulative_kwh = cumulative_joules / 3.6e6
        return {
            "joules": joules,
            "kwh": kwh,
            "seconds": seconds,
            "cumulative_joules": cumulative_joules,
            "cumulative_kwh": cumulative_kwh,
            "cumulative_seconds": cumulative_seconds,
        }

    def print_stats(self, prefix=""):
        stats = self.get_stats()
        print(
            f"{prefix} Energy: {stats['joules']:.2f} J ({stats['kwh']:.6f} kWh) | "
            f"Time: {stats['seconds']:.1f}s | "
            f"Cumulative: {stats['cumulative_joules']:.2f} J ({stats['cumulative_kwh']:.6f} kWh)"
        )
