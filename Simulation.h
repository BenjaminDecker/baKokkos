//
// Created by Benjamin Decker on 03/11/2020.
//

#pragma once

#include "SimulationConfig.h"
#include "LinkedCellsParticleContainer.h"
#include <spdlog//spdlog.h>

/**
 * @brief This class controls the simulation
 */
class Simulation {
 public:
  const SimulationConfig config;
  LinkedCellsParticleContainer container;

  //TODO get those from the ParticlePropertiesLibrary
  const double epsilon = 1;
  const double sigma = 1;
  const double mass = 1;

  const double sigmaPow6 = sigma * sigma * sigma * sigma * sigma * sigma;
  const double twentyFourEpsilonSigmaPow6 = 24 * epsilon * sigmaPow6;
  const double fourtyEightEpsilonSigmaPow12 = twentyFourEpsilonSigmaPow6 * 2 * sigmaPow6;

  explicit Simulation(SimulationConfig config, std::vector<Particle> &particles) : config(std::move(config)) {
    container = LinkedCellsParticleContainer(particles, config);
  }


  /// Starts the simulation
  void start() {
    spdlog::info("Running Simulation...");
    Kokkos::Timer timer;

    //Iteration loop
    for (int iteration = 0; iteration < config.iterations; ++iteration) {
      if (config.vtkFileName) {
        if (iteration % config.vtkWriteFrequency == 0) {
          container.writeVTKFile(iteration, config.iterations, config.vtkFileName.value());
        }
      }

      if (iteration % 1000 == 0) {
        spdlog::info("Iteration: {:0" + std::to_string(std::to_string(config.iterations).length()) + "d}", iteration);
      }
      container.iterateCalculatePositions(config.deltaT);
      container.iterateCalculateForces();
      container.iterateCalculateVelocities(config.deltaT);
    }

    const double time = timer.seconds();
    spdlog::info("Finished simulating. Time: " + std::to_string(time) + " seconds.");
  }
};



