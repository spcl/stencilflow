# Testing suite for distributed program.
# We evaluate different stencils in order to cover different cases
# It indirectly evaluates split_sdfg and SMI integration in DaCe


#!/bin/bash

set -a

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHONPATH=$PYTHONPATH:$SCRIPTPATH/..
PYTHON_BINARY="${PYTHON_BINARY:-python3}"
# where bin utilites are located
BIN_DIR=$SCRIPTPATH/../bin/
# where stencil description files are locate
STENCILS_DIR=$SCRIPTPATH/testing
ERRORS=0
FAILED_TESTS=""
TESTS=0
TEST_TIMEOUT=10

RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'


################################################

bail() {
    ERRORSTR=$1
    /bin/echo -e "${RED}ERROR${NC} in $ERRORSTR" 1>&2
    ERRORS=`expr $ERRORS + 1`
    FAILED_TESTS="${FAILED_TESTS} $ERRORSTR\n"
}

run_jacobi2d() {
    TEST_NAME=jacobi2d_128x128
    TESTS=`expr $TESTS + 1`
    echo -e "${YELLOW}Running test Jacobi 2d...${NC}"

    #0: cleanup
    rm -fr .dacecache/${TEST_NAME}*
    rm *emulated_channel* 2> /dev/null

    #1: Generate the original SDFG and split it into two
    ${BIN_DIR}/sdfg_generator.py ${STENCILS_DIR}/${TEST_NAME}.json ${TEST_NAME}.sdfg
    ${BIN_DIR}/split_sdfg.py ${TEST_NAME}.sdfg b_to_b 0 1 0

    #2: Execute after part and then the before part
    ${BIN_DIR}/run_distributed_program.py ${TEST_NAME}_after.sdfg ${STENCILS_DIR}/${TEST_NAME}.json emulation 1 2 &
    sleep 5;
    ${BIN_DIR}/run_distributed_program.py ${TEST_NAME}_before.sdfg ${STENCILS_DIR}/${TEST_NAME}.json emulation 0 2

    #3: check
    #Save created result
    mv results/${TEST_NAME}/b.dat results/${TEST_NAME}/b_distributed.dat
    # Execute non distributed test
    ${BIN_DIR}/run_program.py ${STENCILS_DIR}/${TEST_NAME}.json emulation
    # diff
    diff results/${TEST_NAME}/b_distributed.dat results/${TEST_NAME}/b.dat

    if [ $? -ne 0 ]; then
        bail "$1 (${RED}Wrong emulation result${NC})"
    fi

    # cleanup
    rm ${TEST_NAME}.sdfg
    rm ${TEST_NAME}_after.sdfg
    rm ${TEST_NAME}_before.sdfg
    rm -fr results/${TEST_NAME}/

    cd -
    return 0
}

run_jacobi3d() {
    TEST_NAME=jacobi3d_32x32x32
    TESTS=`expr $TESTS + 1`
    echo -e "${YELLOW}Running test Jacobi 3d...${NC}"

    #0: cleanup
    rm -fr .dacecache/${TEST_NAME}*
    rm *emulated_channel* 2> /dev/null

    #1: Generate the original SDFG and split it into two
    ${BIN_DIR}/sdfg_generator.py ${STENCILS_DIR}/${TEST_NAME}.json ${TEST_NAME}.sdfg
    ${BIN_DIR}/split_sdfg.py ${TEST_NAME}.sdfg a_to_b 0 1 0

    #2: Execute after part and then the before part
    ${BIN_DIR}/run_distributed_program.py ${TEST_NAME}_after.sdfg ${STENCILS_DIR}/${TEST_NAME}.json emulation 1 2 &
    sleep 5;
    ${BIN_DIR}/run_distributed_program.py ${TEST_NAME}_before.sdfg ${STENCILS_DIR}/${TEST_NAME}.json emulation 0 2

    #3: check
    #Save created result
    mv results/${TEST_NAME}/b.dat results/${TEST_NAME}/b_distributed.dat
    # Execute non distributed test
    ${BIN_DIR}/run_program.py ${STENCILS_DIR}/${TEST_NAME}.json emulation
    # diff
    diff results/${TEST_NAME}/b_distributed.dat results/${TEST_NAME}/b.dat

    if [ $? -ne 0 ]; then
        bail "$1 (${RED}Wrong emulation result${NC})"
    fi

    # cleanup
    rm ${TEST_NAME}.sdfg
    rm ${TEST_NAME}_after.sdfg
    rm ${TEST_NAME}_before.sdfg
    rm -fr results/${TEST_NAME}/

    cd -
    return 0
}


run_jacobi3d_8itr() {
    TEST_NAME=jacobi3d_32x32x32_8itr
    TESTS=`expr $TESTS + 1`
    echo -e "${YELLOW}Running test Jacobi 3d_8itr...${NC}"

    #0: cleanup
    rm -fr .dacecache/${TEST_NAME}*
    rm *emulated_channel* 2> /dev/null

    #1: Generate the original SDFG and split it into two
    ${BIN_DIR}/sdfg_generator.py ${STENCILS_DIR}/${TEST_NAME}.json ${TEST_NAME}.sdfg
    ${BIN_DIR}/split_sdfg.py ${TEST_NAME}.sdfg b6_to_b7 0 1 0

    #2: Execute after part and then the before part
    ${BIN_DIR}/run_distributed_program.py ${TEST_NAME}_after.sdfg ${STENCILS_DIR}/${TEST_NAME}.json emulation 1 2 &
    sleep 5;
    ${BIN_DIR}/run_distributed_program.py ${TEST_NAME}_before.sdfg ${STENCILS_DIR}/${TEST_NAME}.json emulation 0 2

    #3: check
    #Save created result
    mv results/${TEST_NAME}/b7.dat results/${TEST_NAME}/b7_distributed.dat
    # Execute non distributed test
    ${BIN_DIR}/run_program.py ${STENCILS_DIR}/${TEST_NAME}.json emulation
    # diff
    diff results/${TEST_NAME}/b7_distributed.dat results/${TEST_NAME}/b7.dat

    if [ $? -ne 0 ]; then
        bail "$1 (${RED}Wrong emulation result${NC})"
    fi

    # cleanup
    rm ${TEST_NAME}.sdfg
    rm ${TEST_NAME}_after.sdfg
    rm ${TEST_NAME}_before.sdfg
    rm -fr results/${TEST_NAME}/

    cd -
    return 0
}



run_all() {
    run_jacobi2d
    run_jacobi3d
    run_jacobi3d_8itr
}

# Check if aoc is vailable
which aoc
if [ $? -ne 0 ]; then
  echo "aoc not available"
  exit 99
fi

echo "====== Target: INTEL FPGA ======"

DACE_compiler_use_cache=0
DACE_compiler_fpga_vendor="intel_fpga"
DACE_compiler_intel_fpga_board="p520_max_sg280l"

TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $TEST_DIR
run_all ${1:-"0"}

PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi
