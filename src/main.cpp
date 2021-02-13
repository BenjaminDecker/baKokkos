#include <Kokkos_Core.hpp>
#include "Simulation.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include "SimulationConfig/YamlParser.h"

const auto folderName = "Test";

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Simulation simulation = Simulation(SimulationConfig::readConfig(argc, argv));
    simulation.start();
    YamlParser parser(argv[2]);
    std::ofstream outputFile;
    std::filesystem::create_directory(std::to_string(simulation.numParticles) + std::string(" particles"));
    outputFile.open(std::string(folderName) + "/" + std::to_string(parser.stdDev.value()));
    if (!outputFile.is_open()) {
      throw std::runtime_error("");
    }
    outputFile << simulation.time / static_cast<float>(simulation.config.iterations);
    outputFile.close();
    std::cout << "numCells: " << simulation.numCells << "\tlargestCell: " << simulation.largestCell << std::endl;
  }
  Kokkos::finalize();
  return 0;
}