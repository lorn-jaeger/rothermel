# Parallel Project - CSCI 491


> [!NOTE]
> *AI Use Summary*
> No AI use for idea generation, parallelization strategy,
> or implementation of numerical algorithms. I did use
> ChatGPT 5 to take some of my nastier MPI and numpy code
> and show me how to implement them using the libraries as
> intended. Also for quick plotting and printing code.


# Implementation

The serial and parallelized verions of my project are implemented in [serial_levelset.py](./serial_levelset.py) and [mpi_levelset.py](./mpi_levelset.py)

# How to run

I've been using the devcontainer config in your repo for
this class. This code should run there with no problems. The
required libraries are mpi4py, numpy, and matplotlib.


To run the serial version...

```
python serial_levelset.py
```

To run the parallel version...

```
mpirun -n 6 python mpi_levelset.py --nx 3000 --ny 3000 --prows 2 --pcols 3
```

To reproduce my benchmarks...

```
python benchmarks.py
```

# Hardware

I ran all my code either on a Ryzen 5600 with 6 cores or a
64 core Threadripper. 

# Output

Each run should output some .prof and .png files for
profiling and visualization of sim output. 

Somthing like this will be printed to stdout.

```
[Serial]  nx=300, ny=300
[MPI]     nx=300, ny=300, ranks=6

================ Runtime comparison ================
    Grid   Serial (s)      MPI (s)    Speedup
     300        1.558        2.054       0.76
====================================================


========== serial.prof ==========
Wed Oct 22 19:44:22 2025    serial.prof

         57767 function calls (57662 primitive calls) in 1.484 seconds

   Ordered by: cumulative time
   List reduced from 305 to 5 due to restriction <5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.086    0.086    1.484    1.484 /home/lorn/HPC/rothermel/serial_levelset.py:18(main)
     1000    1.190    0.001    1.393    0.001 /home/lorn/HPC/rothermel/serial_levelset.py:33(step)
     4000    0.188    0.000    0.202    0.000 /home/lorn/HPC/.venv/lib/python3.13/site-packages/numpy/_core/numeric.py:1219(roll)
     4000    0.007    0.000    0.010    0.000 /home/lorn/HPC/.venv/lib/python3.13/site-packages/numpy/_core/numeric.py:1420(normalize_axis_tuple)
        1    0.000    0.000    0.006    0.006 /home/lorn/HPC/rothermel/serial_levelset.py:6(parse_args)



========== profile_rank2.prof ==========
Wed Oct 22 19:44:23 2025    profile_rank2.prof

         3390 function calls (3359 primitive calls) in 0.060 seconds

   Ordered by: cumulative time
   List reduced from 169 to 5 due to restriction <5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.008    0.008    0.060    0.060 /home/lorn/HPC/rothermel/mpi_levelset.py:120(main)
      100    0.045    0.000    0.046    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:99(step_upwind)
      100    0.005    0.000    0.005    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:62(halo_exchange)
        1    0.000    0.000    0.001    0.001 /home/lorn/HPC/rothermel/mpi_levelset.py:15(parse_args)
      503    0.001    0.000    0.001    0.000 {method 'copy' of 'numpy.ndarray' objects}



========== profile_rank3.prof ==========
Wed Oct 22 19:44:23 2025    profile_rank3.prof

         3390 function calls (3359 primitive calls) in 0.060 seconds

   Ordered by: cumulative time
   List reduced from 169 to 5 due to restriction <5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.008    0.008    0.060    0.060 /home/lorn/HPC/rothermel/mpi_levelset.py:120(main)
      100    0.045    0.000    0.045    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:99(step_upwind)
      100    0.006    0.000    0.006    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:62(halo_exchange)
        1    0.000    0.000    0.001    0.001 /home/lorn/HPC/rothermel/mpi_levelset.py:15(parse_args)
      503    0.001    0.000    0.001    0.000 {method 'copy' of 'numpy.ndarray' objects}



========== profile_rank0.prof ==========
Wed Oct 22 19:44:24 2025    profile_rank0.prof

         370767 function calls (360786 primitive calls) in 0.521 seconds

   Ordered by: cumulative time
   List reduced from 1813 to 5 due to restriction <5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    523/5    0.009    0.000    0.139    0.028 /home/lorn/HPC/.venv/lib/python3.13/site-packages/matplotlib/text.py:65(_get_text_metrics_with_cache)
        1    0.000    0.000    0.093    0.093 /home/lorn/HPC/.venv/lib/python3.13/site-packages/matplotlib/image.py:1526(imsave)
        1    0.000    0.000    0.093    0.093 /home/lorn/HPC/.venv/lib/python3.13/site-packages/PIL/Image.py:2457(save)
        1    0.000    0.000    0.090    0.090 /home/lorn/HPC/.venv/lib/python3.13/site-packages/PIL/PngImagePlugin.py:1307(_save)
        1    0.000    0.000    0.090    0.090 /home/lorn/HPC/.venv/lib/python3.13/site-packages/PIL/ImageFile.py:629(_save)



========== profile_rank4.prof ==========
Wed Oct 22 19:44:23 2025    profile_rank4.prof

         3390 function calls (3359 primitive calls) in 0.060 seconds

   Ordered by: cumulative time
   List reduced from 169 to 5 due to restriction <5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.007    0.007    0.060    0.060 /home/lorn/HPC/rothermel/mpi_levelset.py:120(main)
      100    0.044    0.000    0.045    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:99(step_upwind)
      100    0.006    0.000    0.007    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:62(halo_exchange)
        1    0.000    0.000    0.001    0.001 /home/lorn/HPC/rothermel/mpi_levelset.py:15(parse_args)
      503    0.001    0.000    0.001    0.000 {method 'copy' of 'numpy.ndarray' objects}



========== profile_rank1.prof ==========
Wed Oct 22 19:44:23 2025    profile_rank1.prof

         3390 function calls (3359 primitive calls) in 0.060 seconds

   Ordered by: cumulative time
   List reduced from 169 to 5 due to restriction <5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.007    0.007    0.060    0.060 /home/lorn/HPC/rothermel/mpi_levelset.py:120(main)
      100    0.046    0.000    0.047    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:99(step_upwind)
      100    0.004    0.000    0.004    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:62(halo_exchange)
        1    0.000    0.000    0.001    0.001 /home/lorn/HPC/rothermel/mpi_levelset.py:15(parse_args)
      503    0.001    0.000    0.001    0.000 {method 'copy' of 'numpy.ndarray' objects}



========== profile_rank5.prof ==========
Wed Oct 22 19:44:23 2025    profile_rank5.prof

         3390 function calls (3359 primitive calls) in 0.060 seconds

   Ordered by: cumulative time
   List reduced from 169 to 5 due to restriction <5>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.008    0.008    0.060    0.060 /home/lorn/HPC/rothermel/mpi_levelset.py:120(main)
      100    0.045    0.000    0.046    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:99(step_upwind)
      100    0.005    0.000    0.005    0.000 /home/lorn/HPC/rothermel/mpi_levelset.py:62(halo_exchange)
        1    0.000    0.000    0.001    0.001 /home/lorn/HPC/rothermel/mpi_levelset.py:15(parse_args)
      503    0.001    0.000    0.001    0.000 {method 'copy' of 'numpy.ndarray' objects}
```

---

# References
[Investigating MPI streams as an alternative to halo
exchange](https://static.epcc.ed.ac.uk/dissertations/hpc-msc/2014-2015/Investigating%20MPI%20streams%20as%20an%20alternative%20to%20halo%20exchange.pdf)

[The Rothermel Surface Fire Spread
Model and Associated Developments:
A Comprehensive Explanation](https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf)

[Modeling wildland fire propagation with level set methods](https://www.fs.usda.gov/psw/publications/4402/mallet.2009.modelingWildlandFirePropagation.pdf)

[mpi4py Docs](https://mpi4py.readthedocs.io/en/stable/)

