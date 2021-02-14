#include <Kokkos_Core.hpp>
#include "Simulation.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include "SimulationConfig/YamlParser.h"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Simulation simulation = Simulation(SimulationConfig::readConfig(argc, argv));
    simulation.start();
    YamlParser parser(argv[2]);
    std::ofstream outputFile;
    const auto initFolderName = "Initialization";
    std::filesystem::create_directory(initFolderName);
    outputFile.open(std::string(initFolderName) + "/" + std::to_string(simulation.numParticles));
    if (!outputFile.is_open()) {
      throw std::runtime_error("");
    }
    outputFile << simulation.initTime;
    outputFile.close();

    const auto runtimeFolderName = "Runtime";
    std::filesystem::create_directory(runtimeFolderName);
    outputFile.open(std::string(runtimeFolderName) + "/" + std::to_string(simulation.numParticles));
    if (!outputFile.is_open()) {
      throw std::runtime_error("");
    }
    outputFile << simulation.runTime / static_cast<float>(simulation.config.iterations);
    outputFile.close();

  }
  Kokkos::finalize();
  return 0;
}