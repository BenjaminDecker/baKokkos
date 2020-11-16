cmake_minimum_required(VERSION 3.16)
project(baKokkos CXX)

set(CMAKE_CXX_STANDARD 17)

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
add_executable(baKokkos src/main.cpp src/Containers/DirectSumParticleContainer.cpp src/Containers/DirectSumParticleContainer.h src/Containers/ParticlePropertiesLibrary.cpp src/Containers/ParticlePropertiesLibrary.h src/Containers/Coord3D.h src/Containers/Particle.h src/Yaml/YamlParser.cpp src/Yaml/YamlParser.h src/Yaml/ParticleGroup.h src/Simulation.h src/Containers/ParticleContainer.h src/Containers/LinkedCellsParticleContainer.cpp src/Containers/LinkedCellsParticleContainer.h src/SimulationConfig.h src/Yaml/ParticleCuboid.h src/Yaml/ParticleSphere.h src/Yaml/CubeClosestPacked.h)
target_link_libraries(baKokkos kokkos yaml-cpp cxxopts spdlog)

include(doxygen.cmake)