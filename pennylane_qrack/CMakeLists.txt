cmake_minimum_required(VERSION 3.20)

project(qrack_device)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(qrack_device SHARED qrack_device.cpp)
target_include_directories(qrack_device PUBLIC /home/iamu/GitHub/catalyst/runtime/include)
set_property(TARGET qrack_device PROPERTY POSITION_INDEPENDENT_CODE ON)

