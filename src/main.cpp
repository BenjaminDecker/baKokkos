
#include <Kokkos_Core.hpp>
#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <optional>
#include "Simulation.h"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // TODO add particles from command line. For now it is only possible to add particles from .yaml files
    Simulation simulation = Simulation(SimulationConfig::readConfig(argc, argv));
    simulation.start();
  }
  Kokkos::finalize();
  return 0;
}
