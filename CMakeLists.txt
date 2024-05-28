cmake_minimum_required(VERSION 3.20)

project(qrack_device)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option (ENABLE_OPENCL "Use OpenCL optimizations" ON)
find_package (OpenCL)
if (NOT OpenCL_FOUND)
    set (ENABLE_OPENCL OFF)
endif (NOT OpenCL_FOUND)

link_directories(/usr/local/lib/qrack /usr/lib/qrack)

add_library(qrack_device SHARED pennylane_qrack/qrack_device.cpp)
if (ENABLE_OPENCL)
    target_link_libraries(qrack_device qrack OpenCL)
else (ENABLE_OPENCL)
    target_link_libraries(qrack_device qrack)
endif (ENABLE_OPENCL)
target_include_directories(qrack_device PUBLIC catalyst/runtime/include)
set_property(TARGET qrack_device PROPERTY POSITION_INDEPENDENT_CODE ON)
install(TARGETS qrack_device LIBRARY DESTINATION pennylane_qrack)

