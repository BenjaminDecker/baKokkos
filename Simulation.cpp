//
// Created by ffbde on 03/11/2020.
//

#include "Simulation.h"
Simulation::Simulation(const SimulationConfig &config) : config(config) {

  YamlParser parser = YamlParser(config.yamlFileName);

  container2 = LinkedCellsParticleContainer(parser);
  //container = DirectSumParticleContainer(YamlParser(config.yamlFileName));
}

void Simulation::start() {
  spdlog::info("Running Simulation...");
  Kokkos::Timer timer;

  //Iteration loop
  for (int iteration = 0; iteration < config.iterations; ++iteration) {
    if (config.vtkOutput) {
      if (iteration % config.vtkWriteFrequency == 0) {
        container2.writeVTKFile(iteration, config.iterations, config.vtkFileName);
      }
    }

    if (iteration % 1000 == 0) {
      spdlog::info("Iteration: {:0" + std::to_string(std::to_string(config.iterations).length()) + "d}", iteration);
    }
    container2.iterateCalculatePositions(config.deltaT);
    container2.iterateCalculateForces();
    container2.iterateCalculateVelocities(config.deltaT);
  }

  const double time = timer.seconds();
  spdlog::info("Finished simulating. Time: " + std::to_string(time) + " seconds.");
}
