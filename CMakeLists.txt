cmake_minimum_required(VERSION 3.16)
project(baKokkos CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        kokkos
        GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
        GIT_TAG "3.2.00"
)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY "https://github.com/jbeder/yaml-cpp.git"
        GIT_TAG "yaml-cpp-0.6.3"
)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY "https://github.com/jarro2783/cxxopts.git"
        GIT_TAG "v2.2.1"
)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
        GIT_TAG "v1.8.1"
)
FetchContent_MakeAvailable(kokkos yaml-cpp cxxopts spdlog)
add_executable(baKokkos src/main.cpp src/ParticleContainers/DirectSum/DirectSumParticleContainer.cpp src/ParticleContainers/DirectSum/DirectSumParticleContainer.h src/Helper/ParticlePropertiesLibrary.cpp src/Helper/ParticlePropertiesLibrary.h src/Helper/Coord3D.cpp src/Helper/Coord3D.h src/Helper/Particle.cpp src/Helper/Particle.h src/Yaml/YamlParser.cpp src/Yaml/YamlParser.h src/Yaml/ParticleGroup.cpp src/Yaml/ParticleGroup.h src/Simulation.cpp src/Simulation.h src/ParticleContainers/ParticleContainer.cpp src/ParticleContainers/ParticleContainer.h src/ParticleContainers/LinkedCells/LinkedCellsParticleContainer.cpp src/ParticleContainers/LinkedCells/LinkedCellsParticleContainer.h src/ParticleContainers/LinkedCells/Cell.cpp src/ParticleContainers/LinkedCells/Cell.h)
target_link_libraries(baKokkos kokkos yaml-cpp cxxopts spdlog)

include(doxygen.cmake)