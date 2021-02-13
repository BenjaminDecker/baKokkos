
#include <Kokkos_Core.hpp>
#include "Simulation.h"
#include <iostream>
#include <fstream>
#include "SimulationConfig/YamlParser.h"

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int numParticles;
    std::vector<double> times;
    for (int i = 0; i < 10; ++i) {
      Simulation simulation = Simulation(SimulationConfig::readConfig(argc, argv));
      numParticles = simulation.numParticles;
      simulation.start();
      times.push_back(simulation.time);
    }
    double acc = 0;
    for(double d : times) {
      acc += d;
    }
    acc /= times.size();

    YamlParser parser(argv[2]);
    std::ofstream outputFile;
    outputFile.open("numParticles: " + std::to_string(numParticles) + "\tIterations: " + std::to_string(parser.iterations.value()));
    if (!outputFile.is_open()) {
      throw std::runtime_error("");
    }
    outputFile << acc << std::endl;
    outputFile.close();
    std::cout << "Time: " <<  acc << " seconds" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
