#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -t 12:00:00  # time requested in hour:minute:second
#SBATCH -p long
#SBATCH -A hpc-prf-cosmo
#SBATCH --mail-type all
#SBATCH --mail-user kustera@ethz.ch

module load intelFPGA_pro/18.1.1_max nalla_pcie/18.1.1_max
aoc -v -board=p520_max_sg280l channels.cl -global-ring -duplicate-ring -o channels.aocx

