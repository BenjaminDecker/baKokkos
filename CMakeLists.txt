cmake_minimum_required(VERSION 3.16)
project(baKokkos CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Kokkos REQUIRED)

add_executable(baKokkos main.cpp Particle.cpp Particle.h ParticleContainer.cpp ParticleContainer.h ParticlePropertiesLibrary.cpp ParticlePropertiesLibrary.h)
target_link_libraries(baKokkos Kokkos::kokkos)