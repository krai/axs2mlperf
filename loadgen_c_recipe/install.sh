#! /bin/bash

function exit_if_error() {
  if [ "${?}" != "0" ]; then
    echo ""
    echo "ERROR: $1"
    exit 1
  fi
}

############################################################
echo ""
echo "Configuring CMake ..."

mkdir -p ${INSTALL_DIR}/install
mkdir -p ${INSTALL_DIR}/obj
cd ${INSTALL_DIR}/obj

_CMAKE_CMD="cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -B . -S ${SRC_DIR}"
echo "${_CMAKE_CMD}"
${_CMAKE_CMD}

exit_if_error "Failed to configure CMake!"

############################################################
_NUM_OF_PROCESSOR="$(nproc)" # TODO: add option to override
echo ""
echo "Building package with make using ${_NUM_OF_PROCESSOR} threads ..."
make -j ${_NUM_OF_PROCESSOR}

exit_if_error "Failed to build package!"

############################################################
echo ""
echo "Installing package ..."

rm -rf install
make install

exit_if_error "Failed to install package!"

############################################################
echo ""
echo "Cleaning obj directory ..."

cd ${INSTALL_DIR}
rm -rf obj