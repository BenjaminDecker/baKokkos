cmake_minimum_required(VERSION 3.16)
project(baKokkos CXX)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(kokkos)
include_directories(${Kokkos_INCLUDE_DIRS_RET})

add_executable(baKokkos main.cpp ParticleContainer.cpp ParticleContainer.h ParticlePropertiesLibrary.cpp ParticlePropertiesLibrary.h)
target_link_libraries(baKokkos kokkos)