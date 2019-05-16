#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 4      # tasks requested
#SBATCH --cpus-per-task=4 
#SBATCH --mem=4096  # memory in Mb
#SBATCH -o exec-outfile  # send stdout to outfile
#SBATCH -e exec-errfile  # send stderr to errfile
#SBATCH -t 00:30:00  # time requested in hour:minute:second
#SBATCH --partition=fpga
#SBATCH --constraint=18.1.1_max
#SBATCH -A hpc-prf-cosmo
#SBATCH --mail-type all
#SBATCH --mail-user kustera@ethz.ch

module load intelFPGA_pro/18.1.1_max nalla_pcie/18.1.1_max
./channel
