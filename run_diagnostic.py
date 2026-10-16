"""
NMRTA Diagnostic Verification Suite & Benchmark Runner.
Validates the core architectural claims from Section 7 of the paper:
1. Efference Copy Cancellation & Sensor Noise Baseline
2. Factorial Perturbation Gating Curve & Detection Rate
3. Global Workspace Bottleneck Compression & Statistical Non-Inferiority
4. Real-Time Hardware Execution Latency Profiling (100 Hz Motor & 10 Hz Cognitive)

Runs live component checks and invokes the complete statistical analysis suite over raw logs.
"""

import os
import sys
import time
import numpy as np
import importlib.util

# Ensure direct file lookup on virtual/9p mounts
class DirectFileFinder:
    def __init__(self, dirpath):
        self.dirpath = dirpath
    def find_spec(self, fullname, path, target=None):
        filename = fullname.split('.')[-1] + '.py'
        filepath = os.path.join(self.dirpath, filename)
        if os.path.isfile(filepath):
            return importlib.util.spec_from_file_location(fullname, filepath)
        return None

_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.meta_path.insert(0, DirectFileFinder(_cur_dir))
sys.path.insert(0, _cur_dir)

from nmrta_system import NMRTASystem
import analyze_results

def run_benchmarks():
    print("=" * 76)
    print("      NMRTA REFERENCE COGNITIVE ARCHITECTURE: DIAGNOSTIC POC SUITE      ")
    print("=" * 76)
    
    # 1. Live component verification
    print("\n[Step 1/3] Pretraining Cerebellar Forward Model & Testing Nominal Gating...")
    system = NMRTASystem(seed=42)
    system.train_cerebellar_baseline(num_steps=2000)
    system.reset(target=[1.0, 1.0])
    system.run_trial(duration=1.0, perturbation_schedule=[])
    
    eps_nom = np.array(system.history["efference_error"])
    print(f"      Nominal Efference Residual: {np.mean(eps_nom):.4f} +/- {np.std(eps_nom):.4f} rad/s")
    print(f"      Reflex Threshold:           tau_refl = 0.1200 rad/s")
    print(f"      Voluntary False Trips:      {np.sum(system.history['reflex_tripped'])} / {len(eps_nom)} steps")
    
    # 2. Live perturbation pulse check
    print("\n[Step 2/3] Testing Force Perturbation Pulse Injection (10 N, 20 ms)...")
    system.reset(target=[1.2, 0.8])
    # Apply a 10 N pulse for 20 ms at t = 0.50 s
    p_sched = [(0.50, 0.52, [0.0, 10.0])]
    system.run_trial(duration=1.0, perturbation_schedule=p_sched)
    
    times = np.array(system.history["time"])
    tripped = np.array(system.history["reflex_tripped"])
    errs = np.array(system.history["efference_error"])
    pulse_window = (times >= 0.50) & (times <= 0.53)
    pulse_detected = np.any(tripped[pulse_window])
    peak_err = np.max(errs[pulse_window]) if np.any(pulse_window) else 0.0
    print(f"      10 N Pulse Detected:        {pulse_detected} (Peak error: {peak_err:.4f} rad/s)")
    
    # 3. Independent statistical analysis over complete experimental dataset
    print("\n[Step 3/3] Running Full Factorial Statistical Analysis & Verification Suite...")
    analyze_results.main()

if __name__ == "__main__":
    run_benchmarks()
