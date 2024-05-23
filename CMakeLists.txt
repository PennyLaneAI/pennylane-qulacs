cmake_minimum_required(VERSION 3.20)

project(qrack_device)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

link_directories(/usr/local/lib/qrack /usr/lib/qrack)

add_library(qrack_device SHARED pennylane_qrack/qrack_device.cpp)
target_link_libraries(qrack_device qrack OpenCL)
target_include_directories(qrack_device PUBLIC /home/iamu/GitHub/catalyst/runtime/include)
set_property(TARGET qrack_device PROPERTY POSITION_INDEPENDENT_CODE ON)
install(TARGETS qrack_device LIBRARY DESTINATION pennylane_qrack)

