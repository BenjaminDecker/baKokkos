cmake_minimum_required(VERSION 3.16)
project(baKokkos)

set(CMAKE_CXX_STANDARD 20)

add_executable(baKokkos main.cpp)
find_package(Kokkos)
target_link_libraries(baKokkos Kokkos::kokkos)