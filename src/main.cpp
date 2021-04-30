#include <Kokkos_Core.hpp>
#include "Simulation.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include "SimulationConfig/YamlParser.h"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // The simulation is created by reading the config information and creating and filling all necessary data structures
    Simulation simulation = Simulation(SimulationConfig::readConfig(argc, argv));

    // The simulation loop is started
    simulation.start();
  }
  Kokkos::finalize();
  return 0;
}