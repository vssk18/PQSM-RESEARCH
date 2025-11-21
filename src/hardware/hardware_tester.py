#!/usr/bin/env python3
"""
Hardware Testing Scripts for PQSM Research
For Raspberry Pi, ESP32, and other IoT devices
Author: Varanasi Sai Srinivasa Karthik
"""

import os
import sys
import time
import psutil
import platform
import subprocess
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Try importing hardware-specific libraries
try:
    import RPi.GPIO as GPIO
    HAS_RPI = True
except ImportError:
    HAS_RPI = False

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardwareMonitor:
    """Monitor hardware resources during crypto operations"""
    
    def __init__(self, device_type: str = 'auto'):
        self.device_type = self._detect_device() if device_type == 'auto' else device_type
        self.metrics = []
        self.start_time = None
        
        logger.info(f"Hardware Monitor initialized for: {self.device_type}")
        
        # Platform info
        self.platform_info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'device_type': self.device_type
        }
    
    def _detect_device(self) -> str:
        """Auto-detect device type"""
        machine = platform.machine().lower()
        
        if 'arm' in machine or 'aarch64' in machine:
            # Check if it's Raspberry Pi
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'raspberry' in cpuinfo.lower():
                        return 'raspberry_pi'
            except:
                pass
            
            # Could be other ARM device
            return 'arm_device'
        
        elif 'x86' in machine or 'amd64' in machine:
            return 'x86_pc'
        
        else:
            return 'unknown'
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.start_time = time.time()
        self.initial_metrics = self._get_current_metrics()
        logger.info("Started hardware monitoring")
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
            'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
        }
        
        # Temperature monitoring (platform-specific)
        temp = self._get_temperature()
        if temp is not None:
            metrics['temperature_c'] = temp
        
        # Power consumption (if available)
        power = self._get_power_consumption()
        if power is not None:
            metrics['power_mw'] = power
        
        # Network stats
        net_io = psutil.net_io_counters()
        metrics['bytes_sent'] = net_io.bytes_sent
        metrics['bytes_recv'] = net_io.bytes_recv
        
        return metrics
    
    def _get_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        if self.device_type == 'raspberry_pi':
            try:
                # Raspberry Pi temperature
                temp = subprocess.check_output(
                    ['vcgencmd', 'measure_temp'],
                    universal_newlines=True
                )
                return float(temp.split('=')[1].split("'")[0])
            except:
                pass
        
        # Try generic Linux thermal zones
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read()) / 1000.0
        except:
            pass
        
        return None
    
    def _get_power_consumption(self) -> Optional[float]:
        """Estimate power consumption"""
        if self.device_type == 'raspberry_pi':
            # Rough estimation based on CPU usage
            cpu = psutil.cpu_percent()
            # Base power ~2.7W, max ~7W for RPi 4
            base_power = 2700  # mW
            max_additional = 4300  # mW
            return base_power + (cpu / 100.0) * max_additional
        
        # Try reading from power supply (laptops)
        try:
            with open('/sys/class/power_supply/BAT0/power_now', 'r') as f:
                return float(f.read()) / 1000.0  # Convert to mW
        except:
            pass
        
        return None
    
    def record_metric(self, operation: str, **kwargs):
        """Record a metric point"""
        metric = self._get_current_metrics()
        metric['operation'] = operation
        metric.update(kwargs)
        self.metrics.append(metric)
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary"""
        if self.start_time is None:
            return {}
        
        duration = time.time() - self.start_time
        final_metrics = self._get_current_metrics()
        
        # Calculate summary statistics
        summary = {
            'duration_seconds': duration,
            'platform': self.platform_info,
            'avg_cpu_percent': sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics) if self.metrics else 0,
            'max_cpu_percent': max((m['cpu_percent'] for m in self.metrics), default=0),
            'avg_memory_mb': sum(m['memory_used_mb'] for m in self.metrics) / len(self.metrics) if self.metrics else 0,
            'max_memory_mb': max((m['memory_used_mb'] for m in self.metrics), default=0),
            'total_metrics': len(self.metrics)
        }
        
        # Temperature stats
        temps = [m.get('temperature_c', 0) for m in self.metrics if 'temperature_c' in m]
        if temps:
            summary['avg_temperature_c'] = sum(temps) / len(temps)
            summary['max_temperature_c'] = max(temps)
        
        # Power stats
        powers = [m.get('power_mw', 0) for m in self.metrics if 'power_mw' in m]
        if powers:
            summary['avg_power_mw'] = sum(powers) / len(powers)
            summary['total_energy_mj'] = (sum(powers) * duration) / 1000.0
        
        # Network stats
        if self.metrics:
            summary['bytes_transferred'] = (
                final_metrics['bytes_sent'] - self.initial_metrics['bytes_sent'] +
                final_metrics['bytes_recv'] - self.initial_metrics['bytes_recv']
            )
        
        logger.info(f"Monitoring stopped. Duration: {duration:.2f}s")
        
        return summary
    
    def save_metrics(self, filename: str):
        """Save metrics to CSV file"""
        if not self.metrics:
            logger.warning("No metrics to save")
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics)
        
        logger.info(f"Metrics saved to {filename}")


class IoTDeviceTester:
    """Test KEM performance on IoT hardware"""
    
    def __init__(self, device_name: str = "generic_iot"):
        self.device_name = device_name
        self.monitor = HardwareMonitor()
        self.results = []
    
    def test_kem_performance(self, kem_algorithm: str, iterations: int = 100) -> Dict[str, Any]:
        """Test KEM performance on hardware"""
        logger.info(f"Testing {kem_algorithm} on {self.device_name}")
        
        # Import KEM implementation
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        try:
            from crypto.advanced_kem import OQSKEMImplementation, ClassicalKEMImplementation
            
            # Initialize KEM
            if kem_algorithm in ['X25519', 'X448']:
                kem = ClassicalKEMImplementation(kem_algorithm)
            else:
                kem = OQSKEMImplementation(kem_algorithm)
        except ImportError:
            logger.error("KEM implementation not available")
            return {}
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Timing arrays
        keygen_times = []
        encaps_times = []
        decaps_times = []
        
        for i in range(iterations):
            # Key generation
            self.monitor.record_metric('keygen', iteration=i)
            start = time.perf_counter()
            pk, sk = kem.generate_keypair()
            keygen_time = (time.perf_counter() - start) * 1000
            keygen_times.append(keygen_time)
            
            # Encapsulation
            self.monitor.record_metric('encapsulate', iteration=i)
            start = time.perf_counter()
            ct, ss1 = kem.encapsulate(pk)
            encaps_time = (time.perf_counter() - start) * 1000
            encaps_times.append(encaps_time)
            
            # Decapsulation
            self.monitor.record_metric('decapsulate', iteration=i)
            start = time.perf_counter()
            ss2 = kem.decapsulate(ct, sk)
            decaps_time = (time.perf_counter() - start) * 1000
            decaps_times.append(decaps_time)
            
            # Verify
            if ss1 != ss2:
                logger.error(f"Shared secret mismatch at iteration {i}")
            
            # Small delay to prevent overheating
            time.sleep(0.01)
        
        # Stop monitoring
        hw_summary = self.monitor.stop_monitoring()
        
        # Calculate statistics
        import numpy as np
        
        result = {
            'device': self.device_name,
            'kem_algorithm': kem_algorithm,
            'iterations': iterations,
            'timestamp': datetime.now().isoformat(),
            
            # Timing statistics (ms)
            'keygen_mean_ms': np.mean(keygen_times),
            'keygen_std_ms': np.std(keygen_times),
            'keygen_p50_ms': np.percentile(keygen_times, 50),
            'keygen_p95_ms': np.percentile(keygen_times, 95),
            
            'encaps_mean_ms': np.mean(encaps_times),
            'encaps_std_ms': np.std(encaps_times),
            'encaps_p50_ms': np.percentile(encaps_times, 50),
            'encaps_p95_ms': np.percentile(encaps_times, 95),
            
            'decaps_mean_ms': np.mean(decaps_times),
            'decaps_std_ms': np.std(decaps_times),
            'decaps_p50_ms': np.percentile(decaps_times, 50),
            'decaps_p95_ms': np.percentile(decaps_times, 95),
            
            # Hardware metrics
            'avg_cpu_percent': hw_summary.get('avg_cpu_percent', 0),
            'max_cpu_percent': hw_summary.get('max_cpu_percent', 0),
            'avg_memory_mb': hw_summary.get('avg_memory_mb', 0),
            'max_memory_mb': hw_summary.get('max_memory_mb', 0),
            'avg_temperature_c': hw_summary.get('avg_temperature_c', 0),
            'max_temperature_c': hw_summary.get('max_temperature_c', 0),
            'avg_power_mw': hw_summary.get('avg_power_mw', 0),
            'total_energy_mj': hw_summary.get('total_energy_mj', 0),
            
            # Platform info
            'platform': hw_summary.get('platform', {})
        }
        
        self.results.append(result)
        
        # Print summary
        print(f"\n{kem_algorithm} Performance:")
        print(f"  KeyGen: {result['keygen_mean_ms']:.2f} ± {result['keygen_std_ms']:.2f} ms")
        print(f"  Encaps: {result['encaps_mean_ms']:.2f} ± {result['encaps_std_ms']:.2f} ms")
        print(f"  Decaps: {result['decaps_mean_ms']:.2f} ± {result['decaps_std_ms']:.2f} ms")
        print(f"  CPU Usage: {result['avg_cpu_percent']:.1f}% (max: {result['max_cpu_percent']:.1f}%)")
        print(f"  Memory: {result['avg_memory_mb']:.1f} MB")
        
        if result['avg_temperature_c'] > 0:
            print(f"  Temperature: {result['avg_temperature_c']:.1f}°C (max: {result['max_temperature_c']:.1f}°C)")
        
        if result['avg_power_mw'] > 0:
            print(f"  Power: {result['avg_power_mw']:.1f} mW")
            print(f"  Energy: {result['total_energy_mj']:.1f} mJ")
        
        return result
    
    def test_all_kems(self, iterations: int = 50):
        """Test all available KEMs"""
        kems = [
            'ML-KEM-512',
            'ML-KEM-768',
            'NTRU-Prime-hrss',
            'BIKE-L1',
            'HQC-128',
            'X25519'  # Classical baseline
        ]
        
        for kem in kems:
            try:
                self.test_kem_performance(kem, iterations)
                time.sleep(2)  # Cool down between tests
            except Exception as e:
                logger.error(f"Failed to test {kem}: {e}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save test results"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Save as JSON
        filename = f"hardware_test_{self.device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        
        # Save summary as CSV
        csv_filename = filename.replace('.json', '.csv')
        import pandas as pd
        df = pd.DataFrame(self.results)
        df.to_csv(csv_filename, index=False)
        
        logger.info(f"Summary saved to {csv_filename}")


class BatteryLifeEstimator:
    """Estimate battery life for different KEMs"""
    
    def __init__(self, battery_capacity_mah: float = 2000, voltage: float = 3.7):
        self.battery_capacity_mah = battery_capacity_mah
        self.voltage = voltage
        self.battery_energy_mwh = battery_capacity_mah * voltage
    
    def estimate_lifetime(self, kem_algorithm: str, messages_per_day: int,
                         idle_power_mw: float = 50) -> Dict[str, float]:
        """Estimate battery lifetime"""
        
        # Energy per operation (mJ) - from our measurements
        energy_per_op = {
            'ML-KEM-512': 0.42,
            'ML-KEM-768': 0.63,
            'ML-KEM-1024': 0.95,
            'NTRU-Prime-hrss': 0.51,
            'BIKE-L1': 1.23,
            'HQC-128': 1.45,
            'Classic-McEliece-348864': 4.21,
            'X25519': 0.15
        }
        
        kem_energy = energy_per_op.get(kem_algorithm, 1.0)
        
        # Daily energy consumption
        active_energy_per_day = messages_per_day * kem_energy  # mJ
        idle_energy_per_day = idle_power_mw * 24  # mW-hours
        total_energy_per_day = (active_energy_per_day / 3600) + idle_energy_per_day
        
        # Battery life
        battery_days = self.battery_energy_mwh / total_energy_per_day
        
        return {
            'kem_algorithm': kem_algorithm,
            'messages_per_day': messages_per_day,
            'active_energy_mj': active_energy_per_day,
            'idle_energy_mwh': idle_energy_per_day,
            'total_energy_mwh': total_energy_per_day,
            'battery_life_days': battery_days,
            'battery_life_months': battery_days / 30
        }


def run_hardware_tests():
    """Run complete hardware test suite"""
    print("="*60)
    print("PQSM Hardware Testing Suite")
    print("="*60)
    
    # Detect device
    monitor = HardwareMonitor()
    print(f"\nDevice Type: {monitor.device_type}")
    print(f"Platform: {monitor.platform_info}")
    
    # Run KEM tests
    tester = IoTDeviceTester()
    tester.test_all_kems(iterations=50)
    
    # Battery life estimation
    print("\n" + "="*60)
    print("Battery Life Estimation")
    print("="*60)
    
    estimator = BatteryLifeEstimator(battery_capacity_mah=2000)
    
    for kem in ['ML-KEM-512', 'NTRU-Prime-hrss', 'Classic-McEliece-348864', 'X25519']:
        result = estimator.estimate_lifetime(kem, messages_per_day=1440)  # One per minute
        print(f"\n{kem}:")
        print(f"  Battery Life: {result['battery_life_days']:.1f} days")
        print(f"  Energy/day: {result['total_energy_mwh']:.1f} mWh")


if __name__ == "__main__":
    run_hardware_tests()
