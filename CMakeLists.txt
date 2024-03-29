cmake_minimum_required(VERSION 3.16)
project(baKokkos CXX)

set(CMAKE_CXX_STANDARD 17)

option(USE_AOS "Use AoS structure for the cells instead of SoA" OFF)

if (USE_AOS)
    add_definitions(-DUSE_AOS)
endif()

include(FetchContent)
FetchContent_Declare(
        kokkos
        GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
        GIT_TAG "3.3.01"
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

file(
        GLOB_RECURSE
        baKokkos_SRC
        "src/*.cpp"
        "src/*.h"
)
add_executable(baKokkos ${baKokkos_SRC})
target_link_libraries(baKokkos kokkos yaml-cpp cxxopts spdlog)

include(doxygen.cmake)