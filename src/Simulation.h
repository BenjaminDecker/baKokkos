//
// Created by Benjamin Decker on 03/11/2020.
//

#pragma once

#include "SimulationConfig/SimulationConfig.h"
#include "LinkedCellsParticleContainer.h"
#include <spdlog//spdlog.h>

/**
 * @brief This class controls the simulation
 */
class Simulation {
 public:
  const SimulationConfig config;
  LinkedCellsParticleContainer container;

  explicit Simulation(const SimulationConfig &config)
      : config(config), container(LinkedCellsParticleContainer(config)) {}

  /// Starts the simulation
  void start() {
    spdlog::info("Running Simulation...");
    Kokkos::Timer timer;

    //Iteration loop
    for (int iteration = 0; iteration < config.iterations; ++iteration) {
      if (iteration % 1000 == 0) {
        spdlog::info("Iteration: {:0" + std::to_string(std::to_string(config.iterations).length()) + "d}", iteration);
      }
      container.doIteration();
    }

    const double time = timer.seconds();
    spdlog::info("Finished simulating. Time: " + std::to_string(time) + " seconds.");
  }
};
