module load intelFPGA_pro/18.1.1_max nalla_pcie/18.1.1_max
g++ -fPIC -g main.cpp utility.cpp `aocl compile-config` `aocl link-config` -o channel 
