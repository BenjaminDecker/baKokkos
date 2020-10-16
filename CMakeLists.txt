cmake_minimum_required(VERSION 3.16)
project(baKokkos)

set(CMAKE_CXX_STANDARD 20)

add_executable(baKokkos main.cpp Particle.cpp Particle.h ParticleContainer.cpp ParticleContainer.h)
find_package(Kokkos)
target_link_libraries(baKokkos Kokkos::kokkos)