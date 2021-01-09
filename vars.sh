# intel fpga
export INTELFPGAOCLSDKROOT=/opt/intelFPGA_pro/19.1/hld
export PATH=$INTELFPGAOCLSDKROOT/bin/:$PATH
export AOCL_BOARD_PACKAGE_ROOT=$INTELFPGAOCLSDKROOT/board/bittware_pcie/s10
# /opt/intelFPGA_pro/19.4/hld/board/bittware_pcie/s10/board_env.xml
# /opt/intelFPGA_pro/19.4/hld/board/bittware_pcie/s10_hpc_default/board_env.xml
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AOCL_BOARD_PACKAGE_ROOT/linux64/lib

# xilinx fpga
export PATH=/opt/Xilinx/Vitis/2019.2/bin:/opt/Xilinx/Vitis_HLS/2019.2/bin:/opt/Xilinx/Vivado/2019.2/bin:$PATH
export XILINX_XRT=/opt/xilinx/xrt
export PATH=$XILINX_XRT/bin:$PATH
export LD_LIBRARY_PATH=$XILINX_XRT/lib:$LD_LIBRARY_PATH
export XILINXD_LICENSE_FILE=2100@sgv-license-01
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

