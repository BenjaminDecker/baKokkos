cmake_minimum_required(VERSION 3.16)
project(baKokkos CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        kokkos
        GIT_REPOSITORY "https://github.com/kokkos/kokkos"
        GIT_TAG "3.2.00"
)
FetchContent_MakeAvailable(kokkos)

add_executable(baKokkos main.cpp ParticleContainer.cpp ParticleContainer.h ParticlePropertiesLibrary.cpp ParticlePropertiesLibrary.h Coord3D.cpp Coord3D.h Particle.cpp Particle.h)
target_link_libraries(baKokkos kokkos)