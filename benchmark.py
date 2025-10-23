import subprocess
import time
import numpy as np
import shutil
import sys
import glob
import pstats


MPI_CORES = 6           
PROWS, PCOLS = 2, 3     
SIZES = [300, 600, 900]  
STEPS = 100
DT = 0.5
DX = DY = 1.0
F = 1.0

SERIAL = "serial_levelset.py"
MPI = "mpi_levelset.py"



def run_serial(nx):
    print(f"\n[Serial]  nx={nx}, ny={nx}")
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, SERIAL,
         "--nx", str(nx), "--ny", str(nx),
         "--dx", str(DX), "--dy", str(DY),
         "--dt", str(DT),
         "--out_prefix", f"serial_{nx}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    return t1 - t0


def run_mpi(nx):
    print(f"[MPI]     nx={nx}, ny={nx}, ranks={MPI_CORES}")
    t0 = time.perf_counter()
    subprocess.run(
        [
            "mpirun", "-n", str(MPI_CORES), sys.executable, MPI,
            "--nx", str(nx), "--ny", str(nx),
            "--dx", str(DX), "--dy", str(DY),
            "--dt", str(DT),
            "--steps", str(STEPS),
            "--out_prefix", f"mpi_{nx}",
            "--prows", str(PROWS), "--pcols", str(PCOLS)
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    t1 = time.perf_counter()
    return t1 - t0

results = []
for n in SIZES:
    t_serial = run_serial(n)
    t_mpi = run_mpi(n)
    speedup = t_serial / t_mpi if t_mpi > 0 else np.nan
    results.append((n, t_serial, t_mpi, speedup))

    print("\n================ Runtime comparison ================")
    print(f"{'Grid':>8s} {'Serial (s)':>12s} {'MPI (s)':>12s} {'Speedup':>10s}")
    for n, ts, tp, s in results:
        print(f"{n:8d} {ts:12.3f} {tp:12.3f} {s:10.2f}")
    print("====================================================\n")

   

    for fname in glob.glob("*.prof"):
        print(f"\n========== {fname} ==========")
        stats = pstats.Stats(fname)
        stats.sort_stats("cumulative").print_stats(5)




